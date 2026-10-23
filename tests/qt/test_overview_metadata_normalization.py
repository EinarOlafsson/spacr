"""Metadata choices and explicit display-only normalization mathematics."""
from types import SimpleNamespace
import threading

import numpy as np
import pytest

from spacr.qt.widgets import qc_field_browser as browser


def _field(plate="p", well="A01", row="A", column=1, number=0):
    return SimpleNamespace(plate=plate, well=well, row=row, column=column,
                           identity=(plate, well, number), files=(("0", "image.tif"),))


def _reader(arrays, calls):
    def read(field, channels, *, edge):
        calls.append((field.identity, tuple(channels), edge))
        return tuple((channel, str(channel), arrays[field.identity][channel]) for channel in channels)
    return read


def test_per_image_normalization_keeps_raw_values_and_handles_constant_nonfinite():
    raw = np.array([[10., 20., np.nan], [30., np.inf, -np.inf]])
    constant = np.full((2, 3), 9, dtype=np.uint16)
    old = raw.copy()
    plan = browser._overview_normalization_plan((), (0, 1), {"percentiles": (0, 100)}, None)
    result = browser._overview_normalized_planes(((0, "a", raw), (1, "b", constant)), plan)
    np.testing.assert_array_equal(result[0][2], [[0., .5, 0.], [1., 0., 0.]])
    np.testing.assert_array_equal(result[1][2], np.zeros((2, 3)))
    np.testing.assert_array_equal(raw, old)
    assert constant.dtype == np.uint16 and np.all(constant == 9)
    assert all(not np.shares_memory(item[2], raw) for item in result)


@pytest.mark.parametrize("mode,reference", [
    ("reference_well", ("p", "A01")),
    ("reference_row", ("p", "A")),
    ("reference_column", ("p", 1)),
    ("reference_condition", "control"),
])
def test_reference_scope_uses_only_explicit_plate_qualified_or_condition_fields(mode, reference):
    selected = _field()
    other_plate = _field("other")
    other_well = _field(well="B02", row="B", column=2)
    fields = (selected, other_plate, other_well)
    arrays = {field.identity: {0: np.array([0., 10.]) if field is selected else np.array([100., 200.])}
              for field in fields}
    calls = []
    options = {"mode": mode, "reference": reference, "percentiles": (0, 100),
               "conditions": {("p", "A01"): "control", ("other", "A01"): "other"}}
    plan = browser._overview_normalization_plan(fields, (0,), options, _reader(arrays, calls))
    assert calls == [(selected.identity, (0,), 256)]
    assert plan["bounds"] == {0: (0., 10.)}
    assert plan["sample_pixels"] == {0: 2}
    assert plan["field_identities"] == (selected.identity,)


def test_shared_bounds_pool_original_finite_pixels_without_mutation():
    fields = (_field(), _field(number=1))
    arrays = {fields[0].identity: {0: np.array([0., 2., np.nan])},
              fields[1].identity: {0: np.array([4., 6., np.inf])}}
    before = {key: value[0].copy() for key, value in arrays.items()}
    plan = browser._overview_normalization_plan(fields, (0,),
        {"mode": "shared", "percentiles": (25, 75)}, _reader(arrays, []))
    assert plan["bounds"] == {0: (1.5, 4.5)}
    assert plan["sample_pixels"] == {0: 4}
    for key in arrays:
        np.testing.assert_array_equal(arrays[key][0], before[key])


@pytest.mark.parametrize("mean_match", [False, True])
def test_reference_channel_bounds_and_mean_matching_precede_same_affine_transform(mean_match):
    field = _field()
    arrays = {field.identity: {0: np.array([10., 20., 30.]), 1: np.array([20., 40., 60.])}}
    plan = browser._overview_normalization_plan((field,), (0, 1),
        {"mode": "shared", "percentiles": (0, 100), "reference_channel": 1,
         "mean_match": mean_match}, _reader(arrays, []))
    assert plan["bounds"] == {0: (20., 60.), 1: (20., 60.)}
    assert plan["gains"] == {0: 2. if mean_match else 1., 1: 1.}
    result = browser._overview_normalized_planes(
        tuple((channel, str(channel), raw) for channel, raw in arrays[field.identity].items()), plan)
    np.testing.assert_array_equal(result[0][2], [0., .5, 1.] if mean_match else [0., 0., .25])
    np.testing.assert_array_equal(result[1][2], [0., .5, 1.])


def test_mean_matching_includes_zero_samples_and_is_not_a_pixel_ratio():
    field = _field()
    arrays = {field.identity: {0: np.array([0., 0., 6.]), 1: np.array([0., 6., 6.])}}
    plan = browser._overview_normalization_plan((field,), (0, 1),
        {"mode": "shared", "percentiles": (0, 100), "reference_channel": 1,
         "mean_match": True}, _reader(arrays, []))
    assert plan["gains"] == {0: 2., 1: 1.}
    assert plan["sample_pixels"] == {0: 3, 1: 3}
    result = browser._overview_normalized_planes(
        tuple((channel, str(channel), raw) for channel, raw in arrays[field.identity].items()), plan)
    np.testing.assert_array_equal(result[0][2], [0., 0., 1.])
    assert np.isfinite(result[0][2]).all()


@pytest.mark.parametrize("values", [[0., 0.], [-1., 0.], [np.nan, np.inf]])
def test_undefined_reference_means_are_refused(values):
    field = _field()
    arrays = {field.identity: {0: np.array(values), 1: np.array([1., 2.])}}
    with pytest.raises(ValueError, match="finite"):
        browser._overview_normalization_plan((field,), (0, 1),
            {"mode": "shared", "reference_channel": 1, "mean_match": True}, _reader(arrays, []))


def test_missing_reference_is_refused_before_any_image_read():
    with pytest.raises(ValueError, match="reference has no fields"):
        browser._overview_normalization_plan((_field(),), (0,),
            {"mode": "reference_well", "reference": ("missing", "A01")}, None)


def test_normalization_samples_are_bounded_original_pixels_and_cancellable():
    fields = tuple(_field(number=number) for number in range(129))
    raw = np.arange(65536, dtype=np.uint16).reshape(256, 256)
    arrays = {field.identity: {0: raw} for field in fields}
    calls = []
    plan = browser._overview_normalization_plan(fields, (0,),
        {"mode": "shared", "percentiles": (0, 100)}, _reader(arrays, calls))
    assert len(calls) == plan["sample_fields"] == 128
    assert plan["sample_pixels"] == {0: 128 * 2048}
    assert calls[0][0] == fields[0].identity and calls[-1][0] == fields[-1].identity
    assert plan["bounds"] == {0: (0., 65535.)}
    assert all(call[2] == 256 for call in calls)
    np.testing.assert_array_equal(raw, np.arange(65536, dtype=np.uint16).reshape(256, 256))
    cancelled = threading.Event()
    cancelled.set()
    from concurrent.futures import CancelledError
    with pytest.raises(CancelledError):
        browser._overview_normalization_plan(fields, (0,), {"mode": "shared"}, None, cancelled)


@pytest.mark.parametrize("key,name,expected", [
    ("cellvoyager", "plate1_E01_T0001F001L01A01Z01C02.tif",
     {"plateID": "plate1", "wellID": "E01", "fieldID": "001", "timeID": "0001",
      "sliceID": "01", "chanID": "02"}),
    ("canonical", "p_A01_F2_C4.tif", {"plateID": "p", "wellID": "A01", "fieldID": "2", "chanID": "4"}),
])
def test_metadata_preview_reuses_generate_masks_presets(key, name, expected):
    from spacr.qt.regex_detect import BUILTIN_REGEXES
    rows, missed = browser._overview_metadata_preview((name, "not-an-image.txt"), BUILTIN_REGEXES[key])
    assert missed == ("not-an-image.txt",)
    assert rows[0][0] == name
    assert {key: rows[0][1][key] for key in expected} == expected
    assert rows[1][1] == {}


def test_custom_regex_preview_uses_actual_filename_captures():
    pattern = r"(?P<plateID>P\d+)-(?P<wellID>[A-Z]\d+)-s(?P<fieldID>\d+)-c(?P<chanID>\d+)\.tif$"
    rows, missed = browser._overview_metadata_preview(("P7-B03-s2-c9.tif",), pattern)
    assert not missed
    assert rows == (("P7-B03-s2-c9.tif", {"plateID": "P7", "wellID": "B03", "fieldID": "2", "chanID": "9"}),)


def test_metadata_preview_matches_reader_case_insensitive_filename_policy():
    from spacr.qt.regex_detect import BUILTIN_REGEXES
    rows, missed = browser._overview_metadata_preview(("p_a01_f2_c4.TIF",), BUILTIN_REGEXES["canonical"])
    assert not missed
    assert rows[0][1] == {"plateID": "p", "wellID": "a01", "fieldID": "2", "chanID": "4"}
