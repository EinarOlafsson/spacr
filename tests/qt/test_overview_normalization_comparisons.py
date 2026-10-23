"""Verify experiment grouping, original-detail crops and annotated PDF parity."""
import json
import threading
from concurrent.futures import CancelledError

import numpy as np
import pytest

from spacr.qt.widgets import qc_field_browser as browser


def _field(tmp_path, well, values, plate="plate1", field="1"):
    path = tmp_path / (plate + well + field + ".npy")
    np.save(path, np.asarray(values))
    return browser._OverviewField(plate, well, well[0], int(well[1:]), field,
                                  (("", str(path)),))


def test_shared_well_limits_and_metadata_comparisons(tmp_path):
    first = _field(tmp_path, "A01", np.arange(16).reshape(4, 4))
    second = _field(tmp_path, "A01", np.arange(16, 32).reshape(4, 4), field="2")
    third = _field(tmp_path, "B02", np.arange(100, 116).reshape(4, 4))
    options = dict(method="minmax", scope="well")
    shared = browser._overview_advanced_normalization((first, second, third), (0,), options)
    image1 = browser._overview_pixels(first, advanced_normalization=shared)
    image2 = browser._overview_pixels(second, advanced_normalization=shared)
    assert image1[-1, -1, 0] < image2[-1, -1, 0] == 255
    assert image2[0, 0, 0] > image1[0, 0, 0] == 0
    assert shared["field_groups"][repr(first.identity)] == shared["field_groups"][repr(second.identity)]
    annotation = {(first.plate, first.well): {"condition": "control"},
                  (third.plate, third.well): {"condition": "control"}}
    shared = browser._overview_advanced_normalization((first, third), (0,),
        dict(method="minmax", scope="condition"), annotation)
    assert len(shared["bounds"]) == 1
    assert browser._overview_pixels(first, advanced_normalization=shared)[-1, -1, 0] < 40
    assert shared["provenance"]["sampled_fields"] == 2
    json.dumps(shared)


def test_manual_background_gamma_and_channel_overrides(tmp_path):
    raw = np.stack((np.array([[10., 60.], [110., np.nan]]),
                    np.array([[100., 150.], [200., 200.]])), axis=-1)
    field = _field(tmp_path, "A01", raw)
    options = dict(method="manual", low=0, high=100, baseline={0: 10, 1: 100},
                   per_channel={"1": dict(gamma=2)})
    pixels = browser._overview_pixels(field, (0, 1), advanced_normalization=options)
    assert pixels[0, 0].tolist() == [0, 0, 0]
    assert pixels[0, 1, 0] == 127
    assert pixels[0, 1, 1] == 180
    assert pixels[1, 0].tolist() == [255, 255, 0]
    assert pixels[1, 1, 0] == 0
    with pytest.raises(ValueError, match="Gamma"):
        browser._overview_pixels(field, advanced_normalization=dict(gamma=0))
    with pytest.raises(ValueError, match="Manual"):
        browser._overview_pixels(field, advanced_normalization=dict(method="manual", low=5, high=1))


@pytest.mark.parametrize("method", ("percentile", "minmax", "manual", "zscore", "log", "asinh", "none"))
def test_full_detail_crop_matches_same_normalized_full_image(tmp_path, method):
    field = _field(tmp_path, "A01", np.arange(120).reshape(10, 12))
    normalization = browser._overview_advanced_normalization((field,), (0,), dict(method=method))
    whole = browser._overview_pixels(field, edge=None, advanced_normalization=normalization)
    crop = browser._overview_pixels(field, edge=None, region=(2, 7, 3, 9), advanced_normalization=normalization)
    assert whole.shape == (10, 12, 3)
    np.testing.assert_array_equal(crop, whole[2:7, 3:9])
    with pytest.raises(ValueError, match="region"):
        browser._overview_pixels(field, region=(-1, 2, 0, 2), advanced_normalization=normalization)


def test_sampling_is_bounded_cancelable_and_finite(tmp_path):
    values = np.arange(1000000, dtype=float).reshape(1000, 1000)
    values[:5] = np.nan
    field = _field(tmp_path, "A01", values)
    result = browser._overview_advanced_normalization((field,), (0,), dict(sample_limit=100))
    limits = next(iter(result["bounds"].values()))["0"]
    assert limits["sample_count"] <= 100
    stopped = threading.Event()
    stopped.set()
    with pytest.raises(CancelledError):
        browser._overview_advanced_normalization((field,), (0,), {}, cancelled=stopped)
    invalid = _field(tmp_path, "B01", np.full((5, 5), np.nan))
    result = browser._overview_advanced_normalization((invalid,), (0,), {})
    assert result["provenance"]["errors"]
    with pytest.raises(ValueError, match="normalization samples"):
        browser._overview_pixels(invalid, advanced_normalization=result)


def test_export_carries_annotations_and_exact_display_normalization(tmp_path, monkeypatch):
    from spacr import plot
    from spacr.figures import bundle

    first = _field(tmp_path, "A01", np.arange(16).reshape(4, 4))
    second = _field(tmp_path, "B01", np.arange(100, 116).reshape(4, 4))
    normalization = browser._overview_advanced_normalization((first, second), (0,),
        dict(method="minmax", scope="plate", per_channel={"0": dict(method="log", baseline=20)}))
    captures = []
    original_register = bundle._register_figure_data
    def register(figure, images, **metadata):
        captures.append((images, metadata, [text.get_text() for text in figure.texts]))
        return original_register(figure, images, **metadata)
    monkeypatch.setattr(bundle, "_register_figure_data", register)
    def save(figure, filename, **kwargs):
        figure.savefig(filename, format="pdf")
    monkeypatch.setattr(plot, "save_figure", save)
    annotations = {("plate1", "A01"): {"condition": "drug", "dose": "10 nM"}}
    destination = tmp_path / "results"
    paths, errors = browser._overview_export((first, second), destination, (0,), (1, 99),
        advanced_normalization=normalization, annotations=annotations)
    assert not errors
    assert len(paths) == 1
    assert (destination / "overview_0001.pdf").read_bytes().startswith(b"%PDF")
    images, metadata, titles = captures[0]
    np.testing.assert_array_equal(images[0], browser._overview_pixels(first, advanced_normalization=normalization))
    assert any("condition: drug" in text and "dose: 10 nM" in text for text in titles)
    assert "0: log" in titles[0]
    assert browser.tr("Subtract background") + "=20.0" in titles[0]
    assert metadata["advanced_normalization"] == normalization
    selection = json.loads((destination / "overview_selection.json").read_text())
    assert selection["fields"][0]["annotations"] == annotations[("plate1", "A01")]
    assert selection["advanced_normalization"] == normalization
    with pytest.raises(ValueError, match="already exist"):
        browser._overview_export((first,), destination, (0,), (1, 99))


@pytest.mark.parametrize("method", ("ratio", "difference"))
def test_reference_normalization_matches_corresponding_pixels_and_roi(tmp_path, method):
    denominator = np.array([[0., 2., 4.], [np.nan, 8., 10.]])
    numerator = np.array([[6., 6., 8.], [9., 12., 10.]])
    field = _field(tmp_path, "A01", np.stack((denominator, numerator), axis=-1))
    options = dict(method=method, reference_channel=0, baseline=1., scope="plate", low=0, high=100)
    model = browser._overview_advanced_normalization((field,), (1,), options)
    diagnostic, = model["provenance"]["reference_diagnostics"]
    assert diagnostic["invalid_reference_count"] == (2 if method == "ratio" else 1)
    assert diagnostic["near_zero_reference_count"] == 1
    actual = browser._overview_pixels(field, (1,), edge=None, advanced_normalization=model)
    crop = browser._overview_pixels(field, (1,), edge=None, advanced_normalization=model, region=(0, 2, 1, 3))
    np.testing.assert_array_equal(crop, actual[:, 1:3])
    with np.errstate(divide="ignore", invalid="ignore"):
        expected = numerator / denominator - 1 if method == "ratio" else numerator - denominator - 1
    low, high = np.nanmin(expected[np.isfinite(expected)]), np.nanmax(expected[np.isfinite(expected)])
    expected = np.uint8(np.clip(np.nan_to_num((expected - low) / (high - low), nan=0, posinf=0), 0, 1) * 255)
    np.testing.assert_array_equal(actual[..., 0], expected)
    assert actual[1, 0].tolist() == [0, 0, 0]


def test_reference_channel_override_is_reflected_in_shared_group_statistics(tmp_path):
    field = _field(tmp_path, "A01", np.stack((np.full((4, 4), 2.), np.arange(16.).reshape(4, 4)), axis=-1))
    options = dict(method="minmax", scope="well", per_channel={"1": dict(method="ratio", reference_channel=0, low=0, high=100)})
    model = browser._overview_advanced_normalization((field,), (0, 1), options)
    limits = next(iter(model["bounds"].values()))
    assert limits["0"]["low"] == limits["0"]["high"] == 2
    assert limits["1"]["low"] == 0 and limits["1"]["high"] == 7.5


def test_many_reference_groups_share_one_global_sample_memory_budget(tmp_path):
    fields = tuple(_field(tmp_path, f"A{index + 1:02d}", np.arange(256).reshape(16, 16)) for index in range(20))
    model = browser._overview_advanced_normalization(fields, (0,), dict(scope="well", method="minmax",
        sample_limit=64, group_sample_limit=64, total_sample_limit=128))
    assert len(model["bounds"]) == 20
    assert sum(limits["0"]["sample_count"] for limits in model["bounds"].values()) <= 128
    assert model["provenance"]["sampled_fields"] == 20


def test_pdf_captions_pixels_and_raw_csv_keep_each_selected_internal_plane(tmp_path, monkeypatch):
    from spacr.figures import bundle
    from spacr import plot, tabular
    fields, selections = [], {}
    for index in range(2):
        values = np.empty((2, 2, 4, 6, 1), np.float32)
        for time in range(2):
            for z in range(2):
                values[time, z, ..., 0] = np.arange(24).reshape(4, 6) + time * 100 + z * 10
        field = _field(tmp_path, "A01", values, field=str(index + 1))
        fields.append(field)
        selections[repr(field.identity)] = dict(time_index=index, z_mode="plane", z_index=1 - index)
    captures = []
    original = bundle._register_figure_data
    def capture(figure, images, **metadata):
        captures.append((images, metadata, [text.get_text() for text in figure.texts]))
        return original(figure, images, **metadata)
    monkeypatch.setattr(bundle, "_register_figure_data", capture)
    monkeypatch.setattr(plot, "save_figure", lambda figure, filename, **kwargs: figure.savefig(filename, format="pdf"))
    destination = tmp_path / "comparison"
    paths, errors = browser._overview_export(tuple(fields), destination, (0,), None,
        measurements=True, quality_edge=None, field_options=selections)
    assert len(paths) == 1 and not errors
    images, metadata, captions = captures[0]
    for index, field in enumerate(fields):
        axes = selections[repr(field.identity)]
        np.testing.assert_array_equal(images[index], browser._overview_pixels(field, (0,), None, edge=None, **axes))
        assert any(f"T={index}" in caption and f"Z={1 - index}" in caption for caption in captions)
        assert metadata["fields"][index]["time_index"] == index
        assert metadata["fields"][index]["z_index"] == 1 - index
    provenance = json.loads((destination / "overview_selection.json").read_text())
    assert provenance["field_options"] == selections
    rows = tabular.read_table(destination / "overview_measurements.csv", canonicalise=False)
    assert rows["channel_0_intensity_mean"].tolist() == [21.5, 111.5]


@pytest.mark.parametrize("mode", ("max", "mean"))
def test_npy_projection_uses_selected_time_native_region_and_reference_channel(tmp_path, mode):
    raw = np.empty((2, 3, 6, 8, 2), np.float32)
    grid = np.arange(48).reshape(6, 8)
    for time in range(2):
        for z in range(3):
            raw[time, z, ..., 0] = grid + time * 100 + z * 10 + 1
            raw[time, z, ..., 1] = (grid + 1) * 2 + time * 200 + z * 10
    field = _field(tmp_path, "A01", raw)
    axes = dict(time_index=1, z_mode=mode, z_index=0)
    plane = next(browser._overview_raw_planes(field, (1,), **axes))[2]
    expected = raw[1, ..., 1].max(axis=0) if mode == "max" else raw[1, ..., 1].mean(axis=0)
    np.testing.assert_array_equal(plane, expected)
    crop = next(browser._overview_raw_planes(field, (1,), region=(1, 4, 2, 6), **axes))[2]
    np.testing.assert_array_equal(crop, expected[1:4, 2:6])
    reader = lambda target, channels: browser._overview_raw_planes(target, channels, **axes)
    model = browser._overview_advanced_normalization((field,), (1,),
        dict(method="difference", reference_channel=0, low=0, high=100), read_planes=reader)
    actual = browser._overview_pixels(field, (1,), None, edge=None, advanced_normalization=model, **axes)
    np.testing.assert_array_equal(browser._overview_pixels(field, (1,), None, edge=None,
        region=(1, 4, 2, 6), advanced_normalization=model, **axes), actual[1:4, 2:6])


@pytest.mark.parametrize("scope", ("row", "column", "missing_metadata"))
def test_shared_bounds_do_not_merge_unannotated_wells_or_physical_plate_addresses(tmp_path, scope):
    fields = (_field(tmp_path, "A01", np.arange(16).reshape(4, 4)),
        _field(tmp_path, "B02", np.arange(16).reshape(4, 4) + 100))
    model = browser._overview_advanced_normalization(fields, (0,), dict(method="minmax", scope=scope))
    assert len(model["bounds"]) == 2
    assert model["field_groups"][repr(fields[0].identity)] != model["field_groups"][repr(fields[1].identity)]


@pytest.mark.parametrize("options,diagnostic", ((dict(method="invalid"), "supported"),
    (dict(method="ratio", epsilon=0), "epsilon"), (dict(method="ratio", reference_channel=-1), "Reference channel"),
    (dict(method="percentile", low=80, high=20), "Percentiles")))
def test_invalid_transform_parameters_never_render_silent_scientific_values(tmp_path, options, diagnostic):
    field = _field(tmp_path, "A01", np.arange(16).reshape(4, 4))
    with pytest.raises(ValueError, match=diagnostic):
        browser._overview_pixels(field, advanced_normalization=options)


def test_ratio_to_self_preserves_invalid_reference_policy_and_zero_range(tmp_path):
    field = _field(tmp_path, "A01", np.array([[0., 2.], [np.nan, 3.]]))
    model = browser._overview_advanced_normalization((field,), (0,), dict(method="ratio", low=0, high=100))
    bounds = next(iter(model["bounds"].values()))["0"]
    assert bounds["low"] == bounds["high"] == 1
    diagnostic, = model["provenance"]["reference_diagnostics"]
    assert diagnostic["invalid_reference_count"] == 2
    assert not browser._overview_pixels(field, advanced_normalization=model).any()


@pytest.mark.parametrize("failure", ("duplicate", "missing", "cancel"))
def test_reference_plans_reject_incomplete_ambiguous_or_cancelled_channel_samples(tmp_path, failure):
    field = _field(tmp_path, "A01", np.ones((4, 6, 2)))
    stopped = threading.Event()
    def reader(target, channels, edge=256):
        if failure == "cancel":
            stopped.set()
        yield 0, target.files[0][1], np.ones((4, 6))
        if failure != "missing":
            yield 0 if failure == "duplicate" else 1, target.files[0][1], np.ones((4, 6))
    error = CancelledError if failure == "cancel" else ValueError
    with pytest.raises(error):
        browser._overview_normalization_plan((field,), (0, 1), dict(mode="shared"), reader, stopped)
    stopped.clear()
    with pytest.raises(error):
        browser._OverviewWellMeanContext((field,), (0, 1), dict(reference=("plate1", "A01"),
            reference_kind="reference_well"), reader, stopped)


def test_advanced_normalization_cancellation_during_reader_and_cached_render_propagates(tmp_path):
    field = _field(tmp_path, "A01", np.arange(24).reshape(4, 6))
    stopped = threading.Event()
    def reader(target, channels):
        stopped.set()
        yield from browser._overview_raw_planes(target, channels)
    with pytest.raises(CancelledError):
        browser._overview_advanced_normalization((field,), (0,), {}, cancelled=stopped, read_planes=reader)
    model = browser._overview_advanced_normalization((field,), (0,), dict(method="minmax"))
    with pytest.raises(CancelledError):
        browser._overview_pixels(field, advanced_normalization=model, cancelled=stopped)


def test_static_channel_is_never_broadcast_into_an_unavailable_stack_plane(tmp_path):
    import tifffile
    stack = tmp_path / "stack.tif"
    static = tmp_path / "static.tif"
    tifffile.imwrite(stack, np.ones((2, 4, 6), np.uint16), metadata={"axes": "TYX"}, photometric="minisblack")
    tifffile.imwrite(static, np.ones((4, 6), np.uint16))
    field = browser._OverviewField("plate1", "A01", "1", 1, "1", (("1", str(stack)), ("2", str(static))))
    with pytest.raises(ValueError, match="no image at the selected"):
        tuple(browser._overview_raw_planes(field, (1, 2), time_index=1))


def test_reference_transform_rejects_mismatched_physical_pixels(tmp_path):
    first = _field(tmp_path, "A01", np.ones((4, 6)))
    reference = tmp_path / "reference.npy"
    np.save(reference, np.ones((3, 5)))
    field = browser._OverviewField("plate1", "A01", "1", 1, "1", (("1", first.files[0][1]), ("0", str(reference))))
    model = browser._overview_advanced_normalization((field,), (1,), dict(method="ratio", reference_channel=0))
    assert "same image size" in model["provenance"]["errors"][0][1]
    assert not model["bounds"]


def test_ratio_pdf_names_reference_and_epsilon_and_embeds_exact_display_pixels(tmp_path):
    from pypdf import PdfReader
    values = np.stack((np.full((4, 6), 2.), np.arange(24).reshape(4, 6)), axis=-1)
    field = _field(tmp_path, "A01", values)
    model = browser._overview_advanced_normalization((field,), (1,),
        dict(method="ratio", reference_channel=0, epsilon=.01, low=0, high=100))
    paths, errors = browser._overview_export((field,), tmp_path / "ratio", (1,), None,
        quality_edge=None, advanced_normalization=model)
    assert not errors
    page, = PdfReader(paths[0]).pages
    text = page.extract_text()
    assert "1: ratio" in text and "epsilon=0.01" in text
    assert browser.tr("Reference channel (ratio / difference)") + "=0" in text
    embedded, = page.images
    expected = browser._overview_pixels(field, (1,), None, edge=None, advanced_normalization=model)
    np.testing.assert_array_equal(np.asarray(embedded.image.convert("RGB"))[::-1], expected)


def test_raw_measurement_cancel_stops_pdf_before_any_partial_image_page(tmp_path, monkeypatch):
    field = _field(tmp_path, "A01", np.arange(24).reshape(4, 6))
    stopped = threading.Event()
    actual = browser._overview_raw_planes
    def cancelled_read(*args, **kwargs):
        values = tuple(actual(*args, **kwargs))
        stopped.set()
        yield from values
    monkeypatch.setattr(browser, "_overview_raw_planes", cancelled_read)
    destination = tmp_path / "cancelled"
    paths, errors = browser._overview_export((field,), destination, (0,), None,
        measurements=True, cancelled=stopped)
    assert not paths and not errors
    metadata = json.loads((destination / "overview_selection.json").read_text())
    assert metadata["cancelled"] and metadata["written_field_count"] == 0
    assert not tuple(destination.glob("*.pdf")) and not (destination / "overview_measurements.csv").exists()


def test_field_sampling_and_empty_montages_keep_explicit_source_identities(tmp_path):
    first = _field(tmp_path, "A01", np.zeros((4, 6)))
    second = _field(tmp_path, "A01", np.ones((4, 6)), field="2")
    from dataclasses import replace
    unresolved = (replace(first, unresolved=True), replace(second, unresolved=True))
    assert browser._overview_sample(unresolved, 1, 0, "well", False) == unresolved
    assert browser._overview_montage((), (0,), None) == (None, ())
    reader = lambda target, channels, edge=256: browser._overview_raw_planes(target, channels, edge=edge)
    context = browser._OverviewWellMeanContext((second,), (0,), dict(reference=("plate1", "A01"),
        reference_kind="reference_well"), reader)
    outside = _field(tmp_path, "A01", np.ones((4, 6)), plate="plate2")
    with pytest.raises(ValueError, match="outside"):
        context.plan_for(outside)
