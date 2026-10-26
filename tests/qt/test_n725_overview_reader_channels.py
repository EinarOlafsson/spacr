"""Declared image channels and physical axes drive bounded overview reads."""
import json

import numpy as np
import pytest
import tifffile

from spacr.qt.widgets import qc_field_browser as browser


@pytest.mark.parametrize("axes", ("YXC", "CYX"))
def test_declared_channel_count_refuses_three_and_preserves_selected_pixels(tmp_path, axes):
    image = np.arange(7 * 9 * 3, dtype=np.uint16).reshape(7, 9, 3)
    stored = image if axes == "YXC" else np.moveaxis(image, -1, 0)
    path = tmp_path / "plate1_A01_F1.tif"
    tifffile.imwrite(path, stored, photometric="rgb" if axes == "YXC" else "minisblack",
                     metadata={"axes": axes})
    field, = browser._overview_index(tmp_path)
    assert browser._overview_channel_info(field)["channels"] == (0, 1, 2)
    plane, = browser._overview_raw_planes(field, (2,))
    np.testing.assert_array_equal(plane[2], image[..., 2])
    with pytest.raises(ValueError, match="Available channels: 0, 1, 2"):
        tuple(browser._overview_raw_planes(field, (3,)))


def test_separate_filename_ids_are_not_rebased_and_duplicates_are_unavailable(tmp_path):
    for channel in (1, 3, 10):
        tifffile.imwrite(tmp_path / f"plate1_A01_F1_C{channel}.tif",
                         np.full((7, 9), channel, np.uint16))
    field, = browser._overview_index(tmp_path)
    assert browser._overview_channel_info(field)["channels"] == (1, 3, 10)
    planes = tuple(browser._overview_raw_planes(field, (10, 1)))
    assert [value[0] for value in planes] == [10, 1]
    np.testing.assert_array_equal(planes[0][2], np.full((7, 9), 10))
    with pytest.raises(ValueError, match="Available channels: 1, 3, 10"):
        tuple(browser._overview_raw_planes(field, (0,)))
    nested = tmp_path / "duplicate"
    nested.mkdir()
    tifffile.imwrite(nested / "plate1_A01_F1_C3.tif", np.ones((7, 9), np.uint16))
    duplicate = browser._OverviewField(field.plate, field.well, field.row, field.column,
                                      field.field, field.files + (("3", str(nested / "plate1_A01_F1_C3.tif")),))
    assert browser._overview_channel_info(duplicate)["channels"] == (1, 10)
    with pytest.raises(ValueError, match="missing or ambiguous"):
        tuple(browser._overview_raw_planes(duplicate, (3,)))


def test_a_filename_id_does_not_silently_choose_an_embedded_rgb_plane(tmp_path):
    path = tmp_path / "plate1_A01_F1_C4.tif"
    tifffile.imwrite(path, np.ones((7, 9, 3), np.uint8), photometric="rgb")
    field, = browser._overview_index(tmp_path)
    assert browser._overview_channel_info(field)["channels"] == ()
    with pytest.raises(ValueError, match="missing or ambiguous"):
        tuple(browser._overview_raw_planes(field, (4,)))


def test_compressed_time_z_channels_read_only_explicit_pages(tmp_path, monkeypatch):
    image = np.arange(2 * 3 * 4 * 7 * 9, dtype=np.uint16).reshape(2, 3, 4, 7, 9)
    path = tmp_path / "plate1_A01_F1.tif"
    tifffile.imwrite(path, image, photometric="minisblack", compression="deflate",
                     metadata={"axes": "TCZYX"})
    field, = browser._overview_index(tmp_path)
    info = browser._overview_channel_info(field)
    assert info["files"][0]["time_count"] == 2
    assert info["files"][0]["z_count"] == 4
    monkeypatch.setattr(tifffile.TiffPageSeries, "asarray", lambda *a, **k: pytest.fail("decoded whole stack"))
    plane, = browser._overview_raw_planes(field, (2,), time_index=1, z_index=3)
    np.testing.assert_array_equal(plane[2], image[1, 2, 3])
    for mode, expected in (("max", image[1, 2].max(axis=0)), ("mean", image[1, 2].mean(axis=0))):
        projected, = browser._overview_raw_planes(field, (2,), time_index=1, z_mode=mode)
        np.testing.assert_array_equal(projected[2], expected)
    with pytest.raises(ValueError, match="explicit time and Z"):
        tuple(browser._overview_raw_planes(field, (2,)))
    with pytest.raises(ValueError, match="valid indices are 0 to 1"):
        tuple(browser._overview_raw_planes(field, (2,), time_index=2, z_index=0))


def test_npy_sampling_keeps_original_stored_values_and_validates_channels(tmp_path):
    image = np.arange(31 * 27 * 3, dtype=np.uint16).reshape(31, 27, 3)
    np.save(tmp_path / "plate1_A01_F1.npy", image)
    field, = browser._overview_index(tmp_path)
    sampled, = browser._overview_raw_planes(field, (1,), edge=8)
    np.testing.assert_array_equal(sampled[2], image[::4, ::4, 1])
    with pytest.raises(ValueError, match="Available channels: 0, 1, 2"):
        tuple(browser._overview_raw_planes(field, (3,)))


def test_filename_axes_remain_typed_and_naturally_ordered_with_explicit_scope(tmp_path):
    pattern = r"(?P<plateID>p1)_(?P<wellID>A01)_F(?P<fieldID>\d+)_T(?P<timeID>\d+)_Z(?P<sliceID>\d+)"
    for value in (10, 2, 1):
        tifffile.imwrite(tmp_path / f"p1_A01_F{value}_T{value}_Z{value}.tif", np.ones((3, 5), np.uint8))
    nested = tmp_path / "derived"
    nested.mkdir()
    tifffile.imwrite(nested / "p1_A01_F99_T1_Z1.tif", np.ones((3, 5), np.uint8))
    fields = browser._overview_index(tmp_path, pattern, source_scope="folder")
    assert [field.field_id for field in fields] == ["1", "2", "10"]
    assert [field.time_id for field in fields] == ["1", "2", "10"]
    assert [field.z_id for field in fields] == ["1", "2", "10"]
    assert fields[1].axes == (("T", "2"), ("Z", "2"))
    assert len(browser._overview_index(tmp_path, pattern)) == 4
    prefixed = tmp_path / "sample-p1_A01_F3_T1_Z1.tif"
    tifffile.imwrite(prefixed, np.ones((3, 5), np.uint8))
    selected = next(field for field in browser._overview_index(tmp_path, pattern, source_scope="folder")
                    if field.files[0][1] == str(prefixed))
    assert selected.unresolved and not selected.well


@pytest.mark.parametrize("channel_count", (1, 4))
def test_full_source_pdf_and_raw_csv_preserve_typed_annotation_provenance(tmp_path, monkeypatch, channel_count):
    from spacr import plot
    image = np.arange(1031 * 517, dtype=np.uint16).reshape(1031, 517)
    path = tmp_path / "plate1_A01_F1.tif"
    stored = image if channel_count == 1 else np.stack([image + index for index in range(4)])
    tifffile.imwrite(path, stored, photometric="minisblack",
                     metadata={"axes": "YX" if channel_count == 1 else "CYX"})
    field, = browser._overview_index(tmp_path)
    captured = []
    original = plot.save_figure

    def save(figure, *args, **kwargs):
        captured.append(([np.asarray(axis.images[0].get_array()).copy()
                          for axis in figure.axes if axis.images], figure._spacr_spec))
        return original(figure, *args, **kwargs)

    monkeypatch.setattr(plot, "save_figure", save)
    output = tmp_path / "export"
    written, errors = browser._overview_export((field,), output, tuple(range(channel_count)), (1, 99),
        measurements=True, quality_edge=None, dpi=150, conditions={(field.plate, field.well): "control"})
    assert len(written) == 1 and not errors
    drawn = captured[0][0]
    assert len(drawn) == channel_count
    if channel_count == 1:
        assert drawn[0].shape == image.shape + (3,)
    else:
        for index, actual in enumerate(drawn):
            plane = stored[index].astype(np.float32)
            low, high = np.percentile(plane, (1, 99))
            expected = np.uint8(np.clip((plane - low) / (high - low), 0, 1) * 255)
            assert actual.shape == image.shape + (3,)
            np.testing.assert_array_equal(actual[..., 0], expected)
        assert captured[0][1]["image_channels"] == [(index,) for index in range(4)]
    assert captured[0][1]["original_resolution"] is True
    from pypdf import PdfReader
    embedded = PdfReader(written[0]).pages[0].images
    assert len(embedded) == channel_count
    assert all(item.image.size == (image.shape[1], image.shape[0]) for item in embedded)
    record = json.loads((output / "overview_selection.json").read_text())["fields"][0]
    assert record["field_id"] == "1" and record["condition"] == "control"
    from spacr import tabular
    measurements = tabular.read_table(output / "overview_measurements.csv")
    assert measurements.loc[0, "condition"] == "control"
    assert measurements.loc[0, "channel_0_intensity_mean"] == pytest.approx(float(image.mean()))


def test_mixed_identified_and_unidentified_channel_files_are_not_silently_dropped(tmp_path):
    first = tmp_path / "plate1_A01_F1_C1.tif"
    second = tmp_path / "plate1_A01_F1.tif"
    for path in (first, second):
        tifffile.imwrite(path, np.ones((3, 5), np.uint8))
    field, = browser._overview_index(tmp_path)
    with pytest.raises(ValueError, match="ambiguous image files"):
        browser._overview_channel_info(field)


def test_sixteen_bit_png_header_and_raw_samples_keep_the_actual_dtype(tmp_path):
    from PIL import Image
    image = np.arange(7 * 9, dtype=np.uint16).reshape(7, 9) * 1000
    path = tmp_path / "plate1_A01_F1.png"
    Image.fromarray(image).save(path)
    field, = browser._overview_index(tmp_path)
    info = browser._overview_channel_info(field)
    assert np.dtype(info["files"][0]["dtype"]) == np.dtype(np.uint16)
    plane, = browser._overview_raw_planes(field, (0,))
    assert plane[2].dtype == np.uint16
    np.testing.assert_array_equal(plane[2], image)


def test_expected_channel_validation_is_a_user_error_without_hiding_programming_faults(tmp_path):
    import traceback
    from spacr import crashreport
    from spacr.errors import ConfigurationError
    path = tmp_path / "plate1_A01_F1.tif"
    tifffile.imwrite(path, np.zeros((3, 5, 3), np.uint8), photometric="rgb")
    field, = browser._overview_index(tmp_path)
    with pytest.raises(ValueError) as caught:
        tuple(browser._overview_raw_planes(field, (3,)))
    assert isinstance(caught.value, ConfigurationError)
    assert crashreport.classify_failure(caught.value) == crashreport.KIND_USER_ERROR
    formatted = "".join(traceback.format_exception(caught.type, caught.value, caught.tb))
    assert crashreport.classify_failure(traceback_text=formatted) == crashreport.KIND_USER_ERROR
    assert crashreport.classify_failure(IndexError("unexpected array indexing")) == crashreport.KIND_FAULT
    assert crashreport.classify_failure(ValueError("unexpected implementation error")) == crashreport.KIND_FAULT


def test_reference_normalization_and_plate_conditions_preserve_raw_export_data(tmp_path, monkeypatch):
    from pypdf import PdfReader
    from spacr import plot, tabular
    from spacr.figures import style

    reference = np.array([[10, 20], [30, 40]], dtype=np.uint16)
    target = np.array([[15, 25], [35, 45]], dtype=np.uint16)
    original_bytes = {}
    for plate, plane in (("p1", reference), ("p2", target)):
        for channel in (1, 2):
            path = tmp_path / f"{plate}_A01_F1_C{channel}.tif"
            tifffile.imwrite(path, plane * channel)
            original_bytes[path] = path.read_bytes()
    fields = browser._overview_index(tmp_path, source_scope="folder")
    assert [(field.plate, field.well) for field in fields] == [("p1", "A01"), ("p2", "A01")]
    conditions = {("p1", "A01"): "reference", ("p2", "A01"): "treated"}
    plan = browser._overview_normalization_plan(fields, (1, 2), {
        "mode": "reference_well", "percentiles": (0, 100),
        "reference": ("p1", "A01"), "reference_channel": 1,
        "mean_match": True, "conditions": conditions,
    }, browser._overview_raw_planes)
    assert plan["bounds"] == {1: (10., 40.), 2: (10., 40.)}
    assert plan["gains"] == {1: 1., 2: .5}
    expected = []
    for plane in (reference, target):
        normalized = np.clip((plane.astype(np.float64) - 10.) / 30., 0., 1.).astype(np.float32)
        rgb = np.zeros(plane.shape + (3,), np.uint8)
        rgb[..., 0] = rgb[..., 1] = np.uint8(normalized * 255)
        expected.append(rgb)
    for field, rgb in zip(fields, expected):
        np.testing.assert_array_equal(browser._overview_pixels(
            field, (1, 2), edge=None, normalization=plan), rgb)
    captured, styled = [], []
    actual_save, actual_style = plot.save_figure, style._apply_user_style

    def apply_style(figure, *args, **kwargs):
        styled.append(figure)
        return actual_style(figure, *args, **kwargs)

    def save(figure, *args, **kwargs):
        assert figure in styled
        drawn = [np.asarray(axis.images[0].get_array()).copy() for axis in figure.axes]
        registered = [np.asarray(image).copy() for image in figure._spacr_image]
        captured.append((drawn, registered, figure._spacr_spec))
        return actual_save(figure, *args, **kwargs)

    monkeypatch.setattr(style, "_apply_user_style", apply_style)
    monkeypatch.setattr(plot, "save_figure", save)
    output = tmp_path / "export"
    written, errors = browser._overview_export(fields, output, (1, 2), (0, 100),
        measurements=True, quality_edge=None, dpi=150,
        normalization=plan, conditions=conditions)
    assert len(written) == 1 and errors == ()
    assert len(captured) == 1
    drawn, registered, spec = captured[0]
    assert spec["original_resolution"] is True and spec["normalization"] == plan
    for actual, data, rgb in zip(drawn, registered, expected):
        np.testing.assert_array_equal(actual, rgb)
        np.testing.assert_array_equal(data, rgb)
    embedded = PdfReader(written[0]).pages[0].images
    assert len(embedded) == 2
    actual_images = [np.asarray(image.image.convert("RGB"))[::-1].copy() for image in embedded]
    assert all(image.shape == (2, 2, 3) for image in actual_images)
    assert {image.tobytes() for image in actual_images} == {image.tobytes() for image in expected}
    table = tabular.read_table(output / "overview_measurements.csv", canonicalise=False)
    for field, raw in zip(fields, (reference, target)):
        row = table.loc[table["plate"] == field.plate].iloc[0]
        assert row["condition"] == conditions[(field.plate, field.well)]
        assert row["field_id"] == 1 and row["well"] == "A01"
        for channel in (1, 2):
            assert row[f"channel_{channel}_intensity_mean"] == float((raw * channel).mean())
            assert row[f"channel_{channel}_source"] == dict(field.files)[str(channel)]
    metadata = json.loads((output / "overview_selection.json").read_text())
    assert metadata["normalization"]["gains"] == {"1": 1., "2": .5}
    assert [(record["plate"], record["well"], record["field_id"], record["condition"])
            for record in metadata["fields"]] == [("p1", "A01", "1", "reference"), ("p2", "A01", "1", "treated")]
    assert metadata["measurements"]["intensity_units"] == "raw_stored_pixel_values"
    assert all(path.read_bytes() == content for path, content in original_bytes.items())


def test_pdf_pages_fit_actual_fields_and_keep_readable_vector_channel_labels(tmp_path, monkeypatch):
    from pypdf import PdfReader
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from spacr import plot

    source = tmp_path / "source"
    source.mkdir()
    for field_id in (1, 2, 3):
        image = np.stack([np.arange(100, dtype=np.uint16).reshape(10, 10) + 10 * channel + field_id
                          for channel in range(4)])
        tifffile.imwrite(source / f"plate1_A01_F{field_id}.tif", image,
                         photometric="minisblack", metadata={"axes": "CYX"})
    fields = browser._overview_index(source, source_scope="folder")
    plan = browser._overview_normalization_plan(fields, (0, 1, 2, 3), {
        "mode": "reference_well", "reference": ("plate1", "A01"),
        "percentiles": (0, 100), "reference_channel": 0, "mean_match": True,
    }, browser._overview_raw_planes)
    actual_save = plot.save_figure
    snapshots = []

    def save(figure, *args, **kwargs):
        canvas = FigureCanvasAgg(figure)
        canvas.draw()
        renderer = canvas.get_renderer()
        snapshots.append((figure, [(axis, image.get_window_extent(renderer))
                                   for axis in figure.axes for image in axis.images],
                          [text.get_text() for text in figure.texts]))
        return actual_save(figure, *args, **kwargs)

    monkeypatch.setattr(plot, "save_figure", save)
    long_condition = "A user condition with several words kept in the PDF caption " * 5
    for count in (1, 3):
        condition = "control" if count == 1 else long_condition
        written, errors = browser._overview_export(fields[:count], tmp_path / f"pdf{count}",
            (0, 1, 2, 3), (0, 100), quality_edge=None, normalization=plan,
            conditions={("plate1", "A01"): condition})
        assert len(written) == 1 and not errors
        figure, images, header = snapshots[-1]
        assert len(images) == count * 4
        assert figure._spacr_spec["grid"] == (1, count)
        for position, (axis, bounds) in enumerate(images):
            assert axis.get_title() == str(position % 4)
            assert axis.title.get_fontsize() >= 10
            title = axis.title.get_window_extent(figure.canvas.get_renderer())
            assert title.y0 >= bounds.y1
            assert title.x0 >= 0 and title.x1 <= figure.bbox.width
            assert title.y1 <= figure.bbox.height
            for other_axis, other_bounds in images:
                if other_axis is not axis:
                    assert not title.overlaps(other_bounds)
        caption_texts = [text for text in figure.texts
                         if text.get_text().startswith("plate1 / A01 /")]
        assert len(caption_texts) == count
        for caption in caption_texts:
            box = caption.get_window_extent(figure.canvas.get_renderer())
            assert box.x0 >= 0 and box.x1 <= figure.bbox.width
            assert box.y0 >= 0 and box.y1 <= figure.bbox.height
            for axis, image_bounds in images:
                assert not box.overlaps(image_bounds)
                assert not box.overlaps(axis.title.get_window_extent(figure.canvas.get_renderer()))
        all_text = "\n".join(header)
        assert browser.tr("Bounds from a reference well") in all_text
        assert "plate1 / A01" in all_text
        assert browser.tr("Mean-matched display scaling before reference bounds") in all_text
        assert browser.tr("On") in all_text
        assert browser.tr("Sampled fields: {fields}; sampled pixels per channel: {pixels}",
                          fields=3, pixels="0: 300, 1: 300, 2: 300, 3: 300") in all_text
        if count == 1:
            occupied = sum(bounds.width * bounds.height for axis, bounds in images)
            assert occupied > .55 * figure.bbox.width * figure.bbox.height
        reader = PdfReader(written[0])
        assert len(reader.pages[0].images) == count * 4
        assert all(item.image.size == (10, 10) for item in reader.pages[0].images)
        text = reader.pages[0].extract_text()
        assert browser.tr("Bounds from a reference well") in text
        assert "plate1 / A01" in text


def test_well_mean_reference_average_preserves_raw_csv_and_full_source_pdf(tmp_path, monkeypatch):
    from pypdf import PdfReader
    from spacr import plot, tabular
    from spacr.figures import style

    source = tmp_path / "source"
    source.mkdir()
    base = np.array([[0, 2], [2, 4]], dtype=np.uint16)
    originals = {}
    for well, field_id, offset in (("A01", 1, 0), ("A01", 2, 4),
                                   ("B01", 1, 8), ("C02", 1, 0)):
        for channel in (1, 2):
            path = source / f"p1_{well}_F{field_id}_C{channel}.tif"
            tifffile.imwrite(path, (base + offset) * channel)
            originals[path] = path.read_bytes()
    fields = browser._overview_index(source, source_scope="folder")
    target, = (field for field in fields if field.well == "C02")
    conditions = {("p1", "A01"): "reference A", ("p1", "B01"): "reference B",
                  ("p1", "C02"): "treated"}
    plan, context = browser._overview_prepare_normalization(fields, (1, 2), {
        "mode": "reference_average", "reference_kind": "reference_column",
        "reference": ("p1", 1), "percentiles": (0, 100), "conditions": conditions,
        "selection": {"seed": 17, "time_index": 0, "z_mode": "plane", "z_index": 0},
    }, browser._overview_raw_planes)
    assert plan["reference_means"] == {1: 7., 2: 14.}
    assert plan["bounds"] == {1: (0., 12.), 2: (0., 24.)}
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    rgb[..., 0] = rgb[..., 1] = np.uint8(np.clip(base * 3.5 / 12., 0., 1.) * 255)
    np.testing.assert_array_equal(browser._overview_pixels(target, (1, 2), edge=None,
        normalization=plan, normalization_context=context), rgb)
    captured, styled = [], []
    actual_save, actual_style = plot.save_figure, style._apply_user_style

    def apply_style(figure, *args, **kwargs):
        styled.append(figure)
        return actual_style(figure, *args, **kwargs)

    def save(figure, *args, **kwargs):
        assert figure in styled
        drawn = [np.asarray(image.get_array()).copy()
                 for axis in figure.axes for image in axis.images]
        captured.append((drawn, [np.asarray(image).copy() for image in figure._spacr_image],
                         figure._spacr_spec))
        return actual_save(figure, *args, **kwargs)

    monkeypatch.setattr(style, "_apply_user_style", apply_style)
    monkeypatch.setattr(plot, "save_figure", save)
    output = tmp_path / "export"
    written, errors = browser._overview_export((target,), output, (1, 2), (0, 100),
        measurements=True, quality_edge=None, dpi=150, normalization=plan,
        normalization_context=context, conditions=conditions)
    assert len(written) == 1 and errors == ()
    drawn, registered, spec = captured[0]
    assert len(captured) == len(drawn) == len(registered) == 1
    np.testing.assert_array_equal(drawn[0], rgb)
    np.testing.assert_array_equal(registered[0], rgb)
    assert spec["original_resolution"] is True
    assert spec["normalization"]["reference_means"] == {1: 7., 2: 14.}
    used, = spec["normalization"]["used_wells"]
    assert (used["plate"], used["well"]) == ("p1", "C02")
    assert used["means"] == {1: 2., 2: 4.} and used["gains"] == {1: 3.5, 2: 3.5}
    assert used["sample_pixels"] == {1: 4, 2: 4}
    assert used["field_identities"] == (target.identity,)
    references = spec["normalization"]["reference_wells"]
    assert [(well["well"], well["means"][1]) for well in references] == [("A01", 4.), ("B01", 10.)]
    page = PdfReader(written[0]).pages[0]
    embedded, = page.images
    assert embedded.image.size == (2, 2)
    np.testing.assert_array_equal(np.asarray(embedded.image.convert("RGB"))[::-1], rgb)
    text = page.extract_text()
    assert browser.tr("Well-mean display scaling to reference average") in text
    assert browser.tr("Reference means: {means}", means="1: 7, 2: 14") in text
    assert "p1 / C02" in text and "3.5" in text
    table = tabular.read_table(output / "overview_measurements.csv", canonicalise=False)
    assert len(table) == 1
    row = table.iloc[0]
    assert (row["plate"], row["well"], row["field_id"], row["condition"]) == ("p1", "C02", 1, "treated")
    for channel in (1, 2):
        assert row[f"channel_{channel}_intensity_mean"] == 2. * channel
        assert row[f"channel_{channel}_source"] == dict(target.files)[str(channel)]
    metadata = json.loads((output / "overview_selection.json").read_text())
    assert metadata["normalization"]["reference_means"] == {"1": 7., "2": 14.}
    assert metadata["normalization"]["used_wells"][0]["gains"] == {"1": 3.5, "2": 3.5}
    assert metadata["fields"][0]["condition"] == "treated"
    assert metadata["measurements"]["original_resolution"] is True
    assert metadata["measurements"]["intensity_units"] == "raw_stored_pixel_values"
    assert all(path.read_bytes() == content for path, content in originals.items())


def test_discovery_refuses_invalid_configuration_without_changing_source_files(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    path = source / "p1_A01_F1.tif"
    tifffile.imwrite(path, np.arange(12, dtype=np.uint16).reshape(3, 4))
    original = path.read_bytes()
    (source / "README.txt").write_text("Source acquisition notes", encoding="utf-8")
    with pytest.raises(browser._OverviewValidationError, match="existing image folder"):
        browser._overview_index(source / "missing")
    with pytest.raises(browser._OverviewValidationError, match="source scope"):
        browser._overview_index(source, source_scope="unknown")
    with pytest.raises(browser._OverviewValidationError, match="unterminated"):
        browser._overview_index(source, pattern="[")
    fields = browser._overview_index(source, source_scope="folder")
    assert len(fields) == 1 and fields[0].files == (("", str(path)),)
    assert path.read_bytes() == original
    assert (source / "README.txt").read_text() == "Source acquisition notes"


def test_declared_unsupported_axes_and_invalid_plane_choices_refuse_without_mutation(tmp_path):
    unsupported = tmp_path / "unsupported"
    unsupported.mkdir()
    unknown = unsupported / "p1_A01_F1.tif"
    tifffile.imwrite(unknown, np.ones((2, 3, 4), np.uint16),
                     photometric="minisblack", metadata={"axes": "QYX"})
    field, = browser._overview_index(unsupported)
    with pytest.raises(browser._OverviewValidationError, match="channel axis is unresolved"):
        browser._overview_channel_info(field)
    source = tmp_path / "supported"
    source.mkdir()
    path = source / "p1_A01_F1.tif"
    image = np.arange(2 * 2 * 3 * 4, dtype=np.uint16).reshape(2, 2, 3, 4)
    tifffile.imwrite(path, image, photometric="minisblack", metadata={"axes": "TZYX"})
    original = {file: file.read_bytes() for file in (unknown, path)}
    field, = browser._overview_index(source)
    for channels, options, message in (
        ((.5,), {}, "nonnegative channel indices"),
        ((-1,), {}, "zero or greater"),
        ((0,), {"z_mode": "median"}, "maximum or mean projection"),
        ((0,), {"time_index": .5, "z_index": 0}, "explicit time and Z"),
    ):
        with pytest.raises(browser._OverviewValidationError, match=message):
            tuple(browser._overview_raw_planes(field, channels, **options))
    channel, actual_path, plane = next(browser._overview_raw_planes(
        field, (0,), time_index=1, z_index=1))
    assert channel == 0 and actual_path == str(path)
    np.testing.assert_array_equal(plane, image[1, 1])
    assert all(file.read_bytes() == content for file, content in original.items())


@pytest.mark.parametrize("method", ("sparse", "full"))
def test_oversized_tiff_uses_bounded_native_mapping_and_refuses_compressed_decode(tmp_path, monkeypatch, method):
    import tracemalloc

    from spacr.errors import ConfigurationError

    monkeypatch.setenv("SPACR_OVERVIEW_DISK_CACHE", "0")
    monkeypatch.setattr(browser, "_overview_read_method", lambda path: method)

    source = tmp_path / "source"
    source.mkdir()
    shape = (8193, 8193)
    mapped_path = source / "p1_A01_F1.tif"
    mapped = tifffile.memmap(mapped_path, shape=shape, dtype=np.uint16,
                            photometric="minisblack")
    mapped[0, 0] = 1234
    mapped[-1, -1] = 4321
    mapped.flush()
    del mapped
    compressed_path = source / "p1_A01_F2.tif"
    tifffile.imwrite(compressed_path, np.zeros(shape, dtype=np.uint16),
                     compression="deflate", photometric="minisblack")
    compressed_bytes = compressed_path.read_bytes()
    mapped_stat = mapped_path.stat()
    fields = browser._overview_index(source)
    first, second = fields
    tracemalloc.start()
    try:
        _, _, preview = next(browser._overview_raw_planes(first, (0,), edge=32))
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert preview.shape == (32, 32) and preview[0, 0] == 1234
    assert peak < 4 << 20 and preview.nbytes == 32 * 32 * 2
    with pytest.raises(ConfigurationError, match="128 MiB preview decode limit"):
        tuple(browser._overview_raw_planes(second, (0,), edge=32))
    assert compressed_path.read_bytes() == compressed_bytes
    assert (mapped_path.stat().st_size, mapped_path.stat().st_mtime_ns) == (
        mapped_stat.st_size, mapped_stat.st_mtime_ns)
    actual = tifffile.memmap(mapped_path, mode="r")
    assert actual[0, 0] == 1234 and actual[-1, -1] == 4321
    del actual


@pytest.mark.parametrize("stage", ("reference-preparation", "full-source-render"))
def test_actual_well_mean_export_cancellation_never_publishes_a_partial_page(tmp_path, stage):
    import threading

    source = tmp_path / "source"
    source.mkdir()
    for well, offset in (("A01", 0), ("B02", 4)):
        tifffile.imwrite(source / f"p1_{well}_F1_C1.tif",
            np.arange(12, dtype=np.uint16).reshape(3, 4) + offset)
    fields = browser._overview_index(source)
    target = next(field for field in fields if field.well == "B02")
    original = {path: path.read_bytes() for path in source.iterdir()}
    cancelled = threading.Event()
    read_fields = []

    def sampled_read(field, channels, *, edge):
        planes = tuple(browser._overview_raw_planes(field, channels, edge=edge))
        read_fields.append(field.identity)
        if field is target and stage == "reference-preparation":
            cancelled.set()
        return planes

    plan, context = browser._overview_prepare_normalization(fields, (1,), {
        "mode": "reference_average", "reference_kind": "reference_well",
        "reference": ("p1", "A01"), "percentiles": (0, 100),
    }, sampled_read, cancelled)
    assert read_fields == [fields[0].identity]
    if stage == "full-source-render":
        context.plan_for(target, cancelled)
        original_read = browser._overview_raw_planes
        from unittest.mock import patch

        def render_read(field, channels, **options):
            for plane in original_read(field, channels, **options):
                yield plane
            if field is target and "edge" in options and options["edge"] is None:
                cancelled.set()

        patcher = patch.object(browser, "_overview_raw_planes", render_read)
    else:
        from contextlib import nullcontext
        patcher = nullcontext()
    output = tmp_path / "export"
    with patcher:
        written, errors = browser._overview_export((target,), output, (1,), (0, 100),
            cancelled=cancelled, measurements=True, quality_edge=None,
            normalization=plan, normalization_context=context)
    assert cancelled.is_set() and written == errors == ()
    assert not list(output.glob("*.pdf"))
    assert not (output / "overview_measurements.csv").exists()
    assert not list(output.glob(".*pending*"))
    metadata = json.loads((output / "overview_selection.json").read_text())
    assert metadata["cancelled"] is True and metadata["fields"] == []
    assert metadata["written_field_count"] == 0 and metadata["requested_field_count"] == 1
    assert metadata["measurements"]["path"] is None
    assert metadata["measurements"]["written_field_count"] == 0
    assert metadata["normalization"]["reference_means"] == {"1": 5.5}
    assert all(path.read_bytes() == content for path, content in original.items())


def test_large_z_mean_refuses_full_float_allocation_but_keeps_raw_stride_preview(tmp_path):
    path = tmp_path / "p1_A01_F1.tif"
    image = tifffile.memmap(path, shape=(2, 4097, 4097), dtype=np.uint8,
                           photometric="minisblack", metadata={"axes": "ZYX"})
    image[0, 0, 0], image[1, 0, 0] = 2, 4
    image.flush()
    del image
    before = path.stat()
    field, = browser._overview_index(tmp_path)
    with pytest.raises(browser._OverviewValidationError, match="128 MiB preview decode limit"):
        tuple(browser._overview_raw_planes(field, (0,), z_mode="mean"))
    _, _, preview = next(browser._overview_raw_planes(field, (0,), z_mode="mean", edge=32))
    assert preview.shape == (32, 32) and preview.dtype == np.float64
    assert preview[0, 0] == 3. and np.count_nonzero(preview) == 1
    assert (path.stat().st_size, path.stat().st_mtime_ns) == (before.st_size, before.st_mtime_ns)
    actual = tifffile.memmap(path, mode="r")
    assert actual[0, 0, 0] == 2 and actual[1, 0, 0] == 4
    del actual


def test_raw_npy_memory_budget_cannot_be_bypassed_by_an_oversized_preview(tmp_path):
    path = tmp_path / "p1_A01_F1.npy"
    image = np.lib.format.open_memmap(path, mode="w+", dtype=np.uint16, shape=(8193, 8193))
    image[0, 0], image[-1, -1] = 1234, 4321
    image.flush()
    del image
    before = path.stat()
    field, = browser._overview_index(tmp_path)
    for edge in (None, 16384):
        with pytest.raises(browser._OverviewValidationError, match="128 MiB preview decode limit"):
            tuple(browser._overview_raw_planes(field, (0,), edge=edge))
    _, actual_path, preview = next(browser._overview_raw_planes(field, (0,), edge=32))
    assert actual_path == str(path) and preview.shape == (32, 32)
    assert isinstance(preview, np.memmap) and preview[0, 0] == 1234
    record = browser._overview_measurement(field, (0,))
    assert record["status"] == "error" and "128 MiB preview decode limit" in record["error"]
    assert record["source_files"] == json.dumps(field.files)
    assert "height_px" not in record and "channel_0_intensity_mean" not in record
    assert (path.stat().st_size, path.stat().st_mtime_ns) == (before.st_size, before.st_mtime_ns)
    actual = np.load(path, mmap_mode="r", allow_pickle=False)
    assert actual[0, 0] == 1234 and actual[-1, -1] == 4321
    del actual


def test_empty_measurement_selection_reports_provenance_without_inventing_a_region(tmp_path):
    path = tmp_path / "p1_A01_F1_C1.tif"
    tifffile.imwrite(path, np.arange(12, dtype=np.uint16).reshape(3, 4))
    original = path.read_bytes()
    field, = browser._overview_index(tmp_path)
    record = browser._overview_measurement(field, (), conditions={("p1", "A01"): "control"})
    assert record["status"] == "error"
    assert record["error"] == browser.tr("Choose one or more nonnegative channel indices.")
    assert record["channels"] == "[]" and record["source_files"] == json.dumps(field.files)
    assert (record["plate"], record["well"], record["field_id"], record["condition"]) == (
        "p1", "A01", "1", "control")
    assert "height_px" not in record and "roi_area" not in record
    assert not any(key.startswith("channel_") for key in record)
    assert path.read_bytes() == original


def test_cancellation_after_actual_raw_measurement_read_does_not_publish_results(tmp_path, monkeypatch):
    import threading

    path = tmp_path / "p1_A01_F1_C1.tif"
    tifffile.imwrite(path, np.arange(12, dtype=np.uint16).reshape(3, 4))
    original_bytes = path.read_bytes()
    field, = browser._overview_index(tmp_path)
    cancelled = threading.Event()
    original_read = browser._overview_raw_planes
    actual_reads = []

    def read_then_cancel(actual_field, channels, **options):
        planes = tuple(original_read(actual_field, channels, **options))
        actual_reads.extend((channel, source, plane.copy()) for channel, source, plane in planes)
        cancelled.set()
        yield from planes

    monkeypatch.setattr(browser, "_overview_raw_planes", read_then_cancel)
    output = tmp_path / "export"
    written, errors = browser._overview_export((field,), output, (1,), (0, 100),
        cancelled=cancelled, measurements=True, quality_edge=None)
    assert len(actual_reads) == 1 and actual_reads[0][:2] == (1, str(path))
    np.testing.assert_array_equal(actual_reads[0][2], np.arange(12).reshape(3, 4))
    assert cancelled.is_set() and written == errors == ()
    assert not list(output.glob("*.pdf")) and not (output / "overview_measurements.csv").exists()
    assert not list(output.glob(".*pending*"))
    metadata = json.loads((output / "overview_selection.json").read_text())
    assert metadata["cancelled"] is True and metadata["fields"] == []
    assert metadata["written_field_count"] == 0 and metadata["requested_field_count"] == 1
    assert metadata["measurements"]["path"] is None
    assert metadata["measurements"]["written_field_count"] == 0
    assert path.read_bytes() == original_bytes


def test_unreadable_field_tiles_keep_neighbors_renderable_and_recover_after_reload(tmp_path, qtbot, monkeypatch):
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QImage
    from PySide6.QtWidgets import QPushButton
    from tests.qt.test_field_of_view_overview import _retire_overview

    source = tmp_path / "source"
    source.mkdir()
    base = np.arange(48 * 64, dtype=np.uint16).reshape(48, 64)
    planes = {"1": base, "2": base[::-1, ::-1].copy(),
              "3": np.where(base % 2, 500, 0).astype(np.uint16),
              "4": base[::-1].copy()}
    paths = {field: source / f"p1_A01_F{field}_C1.tif" for field in planes}
    for field, path in paths.items():
        tifffile.imwrite(path, planes[field])
    originals = {field: path.read_bytes() for field, path in paths.items()}
    dialog = browser._FieldOverview(settings={"source": str(source), "source_scope": "folder",
        "channels": (1,), "geometry": (1, 1), "normalization_percentiles": (0, 100)})

    def retire(widget):
        """Stop this overview's queued visuals before retiring its owned workers."""
        widget._visual_timer.stop()
        _retire_overview(widget, qtbot)

    qtbot.addWidget(dialog, before_close_func=retire)
    messages = []
    actual_status = dialog.status.setText

    def show_status(message):
        """Observe the real status presentation without replacing it."""
        actual_status(message)
        messages.append(dialog.status.text())

    monkeypatch.setattr(dialog.status, "setText", show_status)
    dialog.show()
    dialog._scan()
    qtbot.waitUntil(lambda: len(dialog._fields) == 4 and dialog._normalization_ready
                    and not dialog._index_jobs.is_busy(), timeout=10000)
    indexed = {field.field_id: field for field in dialog._fields}
    paths["2"].write_bytes(b"This acquired TIFF became unreadable after indexing.")
    paths["3"].unlink()

    def open_well():
        """Use the physical well cell to expose its actual indexed field list."""
        well = dialog.wells.item(*dialog._well_positions["A01"])
        qtbot.mouseClick(dialog.wells.viewport(), Qt.LeftButton,
                         pos=dialog.wells.visualItemRect(well).center())
        assert dialog._scope == ("p1", "A01")
        assert ("p1", "A01") in dialog._selected_wells
        qtbot.waitUntil(lambda: dialog.fields.isVisible() and dialog.fields.count() == 4)
        return {dialog.fields.item(index).data(Qt.UserRole).field_id: dialog.fields.item(index)
                for index in range(dialog.fields.count())}

    def wait_for_tile(item):
        """Scroll to the real item and await its worker's installed identity key."""
        dialog.fields.scrollToItem(item)
        qtbot.waitUntil(lambda: item.data(Qt.UserRole + 1) is not None, timeout=10000)

    def open_and_check(item, expected):
        """Click a field and compare its native displayed pixels with the raw transform."""
        dialog.fields.scrollToItem(item)
        qtbot.mouseClick(dialog.fields.viewport(), Qt.LeftButton,
                         pos=dialog.fields.visualItemRect(item).center())
        identity = item.data(Qt.UserRole).identity
        qtbot.waitUntil(lambda: dialog.view._item is not None
                        and dialog._displayed_field[0] == identity
                        and not dialog._image_jobs.is_busy(), timeout=10000)
        image = dialog.view._item.pixmap().toImage().convertToFormat(QImage.Format_RGB888)
        actual = np.frombuffer(image.bits(), dtype=np.uint8).reshape(image.height(), image.bytesPerLine())
        actual = actual[:, :image.width() * 3].reshape(image.height(), image.width(), 3)
        gray = np.uint8(expected.astype(np.float32) / float(expected.max()) * 255)
        np.testing.assert_array_equal(actual, np.repeat(gray[..., None], 3, axis=2))

    try:
        items = open_well()
        for field_id in ("1", "2", "3", "4"):
            wait_for_tile(items[field_id])
        assert not items["1"].icon().isNull() and not items["4"].icon().isNull()
        assert items["2"].icon().isNull() and items["3"].icon().isNull()
        for field_id in ("2", "3"):
            prefix = browser.tr("Could not read {field}: {error}",
                                field=indexed[field_id].caption, error="").rstrip()
            assert any(message.startswith(prefix) for message in messages)
        assert {field: item.data(Qt.UserRole).identity for field, item in items.items()} == {
            field: record.identity for field, record in indexed.items()}
        open_and_check(items["4"], planes["4"])
        assert paths["1"].read_bytes() == originals["1"] and paths["4"].read_bytes() == originals["4"]
        for field_id in ("2", "3"):
            paths[field_id].write_bytes(originals[field_id])
        dialog.options_button.click()
        load = next(button for button in dialog.findChildren(QPushButton)
                    if button.text() == browser.tr("Load overview"))
        dialog._controls_scroll.ensureWidgetVisible(load)
        qtbot.mouseClick(load, Qt.LeftButton)
        qtbot.waitUntil(lambda: len(dialog._fields) == 4 and dialog._normalization_ready
                        and not dialog._index_jobs.is_busy() and not dialog._individual, timeout=10000)
        recovered = open_well()
        for item in recovered.values():
            wait_for_tile(item)
            assert not item.icon().isNull()
        open_and_check(recovered["2"], planes["2"])
        assert all(path.read_bytes() == originals[field] for field, path in paths.items())
    finally:
        retire(dialog)
        dialog.close()


def test_unresolved_sources_remain_distinct_deterministic_samples_in_pdf_and_raw_csv(tmp_path):
    from pypdf import PdfReader
    from spacr import tabular

    source = tmp_path / "unidentified"
    source.mkdir()
    pixels = {
        "capture-alpha.tif": np.arange(30, dtype=np.uint16).reshape(5, 6),
        "capture-beta.npy": np.arange(28, dtype=np.uint16).reshape(4, 7)[::-1].copy(),
        "capture-gamma.npy": np.where(np.indices((6, 5)).sum(axis=0) % 2, 9, 0).astype(np.uint16),
    }
    for name, array in pixels.items():
        if name.endswith(".tif"):
            tifffile.imwrite(source / name, array)
        else:
            np.save(source / name, array)
    original = {path: path.read_bytes() for path in source.iterdir()}
    fields = browser._overview_index(source, source_scope="folder")
    assert len(fields) == len({field.identity for field in fields}) == 3
    assert all(field.unresolved and field.well == field.row == "" and field.column == 0
               for field in fields)
    assert {path for field in fields for channel, path in field.files} == {str(path) for path in original}
    for grouping in ("well", "row", "column", "plate"):
        selected = browser._overview_sample(fields, 1, 37, grouping, True)
        assert selected == browser._overview_sample(tuple(reversed(fields)), 1, 37, grouping, True)
        assert {field.identity for field in selected} == {field.identity for field in fields}
    selected = browser._overview_sample(fields, 1, 37, "well", True)
    output = tmp_path / "export"
    written, errors = browser._overview_export(selected, output, (0,), (0, 100),
        seed=37, measurements=True, quality_edge=None)
    assert len(written) == 1 and errors == ()
    page = PdfReader(written[0]).pages[0]
    assert len(page.images) == len(selected)
    assert {image.image.size for image in page.images} == {(6, 5), (7, 4), (5, 6)}
    text = "".join(page.extract_text().split())
    assert all(field.field in text for field in selected)
    metadata = json.loads((output / "overview_selection.json").read_text())
    assert metadata["seed"] == 37 and metadata["quality_edge"] is None
    assert metadata["requested_field_count"] == metadata["written_field_count"] == 3
    assert [(record["plate"], record["well"], record["field_id"], record["files"])
            for record in metadata["fields"]] == [
        (field.plate, "", field.field_id, [list(item) for item in field.files]) for field in selected]
    table = tabular.read_table(output / "overview_measurements.csv", canonicalise=False,
                               keep_default_na=False)
    assert len(table) == len(selected) and set(table["well"]) == {""}
    assert set(table["row"]) == {""} and set(table["column"]) == {0}
    assert set(table["metadata_resolved"]) == {False} and set(table["status"]) == {"ok"}
    for field in selected:
        row = table.loc[table["field"] == field.field].iloc[0]
        path = next(iter(dict(field.files).values()))
        raw = pixels[field.field]
        assert row["channel_0_source"] == path
        assert row["roi_area"] == raw.size and row["channel_0_intensity_mean"] == float(raw.mean())
    assert all(path.read_bytes() == content for path, content in original.items())


def test_ui_cancellation_during_first_real_pdf_field_discards_partial_outputs(tmp_path, qtbot, monkeypatch, qapp):
    import threading
    from PySide6.QtCore import QThread, QTimer, Qt
    from PySide6.QtWidgets import QFileDialog, QPushButton
    from tests.qt.test_field_of_view_overview import _retire_overview

    source = tmp_path / "source"
    source.mkdir()
    for field_id in (1, 2, 3):
        tifffile.imwrite(source / f"p1_A01_F{field_id}_C1.tif",
            np.arange(48 * 64, dtype=np.uint16).reshape(48, 64) + field_id)
    original_bytes = {path: path.read_bytes() for path in source.iterdir()}
    dialog = browser._FieldOverview(settings={"source": str(source), "source_scope": "folder",
        "channels": (1,), "geometry": (1, 1), "normalization_percentiles": (0, 100)})
    started, release = threading.Event(), threading.Event()
    reads, completed, failures = [], [], []
    actual_pixels = browser._overview_pixels

    def retire(widget):
        """Release this test's reader gate and retire only the overview's jobs."""
        release.set()
        widget._visual_timer.stop()
        _retire_overview(widget, qtbot)

    qtbot.addWidget(dialog, before_close_func=retire)
    dialog.show()
    dialog._scan()
    qtbot.waitUntil(lambda: len(dialog._fields) == 3 and dialog._normalization_ready
                    and not dialog._index_jobs.is_busy(), timeout=10000)
    dialog.options_button.click()
    dialog._controls_scroll.ensureWidgetVisible(dialog.export_scope)
    dialog.export_scope.setCurrentIndex(dialog.export_scope.findData("plate"))
    dialog.measurements.setChecked(True)
    assert dialog.pdf_quality.currentData() is None
    selected = dialog._export_selection()
    assert len(selected) == 3 and all(field.well == "A01" for field in selected)
    assert dialog._normalization_context is None

    def read(field, *args, **kwargs):
        """Pause the real full-source image after decoding, never replace its pixels."""
        pixels = actual_pixels(field, *args, **kwargs)
        if "edge" in kwargs and kwargs["edge"] is None:
            reads.append((field.identity, QThread.currentThread()))
            started.set()
            if not release.wait(10):
                raise RuntimeError("The controlled full-source reader was not released.")
        return pixels

    monkeypatch.setattr(browser, "_overview_pixels", read)
    output = tmp_path / "cancelled-export"
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *args, **kwargs: str(output))
    dialog._export_jobs.job_finished.connect(lambda ok: completed.append((ok, QThread.currentThread())))
    dialog._export_jobs.job_failed.connect(failures.append)
    export = next(button for button in dialog.findChildren(QPushButton)
                  if button.text() == browser.tr("Contact sheet PDFs"))
    cancel = next(button for button in dialog.findChildren(QPushButton)
                  if button.text() == browser.tr("Cancel loading or export"))
    try:
        dialog._controls_scroll.ensureWidgetVisible(export)
        qtbot.mouseClick(export, Qt.LeftButton)
        qtbot.waitUntil(started.is_set, timeout=10000)
        output = next((source / "results").glob("overview_*"))
        assert dialog._export_jobs.is_busy() and len(reads) == 1
        assert reads[0][0] == selected[0].identity
        assert reads[0][1] != qapp.thread()
        heartbeat = []
        QTimer.singleShot(0, lambda: heartbeat.append(QThread.currentThread()))
        qtbot.waitUntil(lambda: bool(heartbeat))
        assert heartbeat == [qapp.thread()]
        dialog._controls_scroll.ensureWidgetVisible(cancel)
        qtbot.mouseClick(cancel, Qt.LeftButton)
        assert dialog._cancelled.is_set() and dialog._export_jobs.is_busy()
        assert dialog.status.text() == browser.tr("Cancellation requested; the current image read may finish.")
        release.set()
        qtbot.waitUntil(lambda: bool(completed) and not dialog._export_jobs.is_busy(), timeout=10000)
        assert completed == [(True, qapp.thread())] and failures == []
        assert len(reads) == 1
        assert not list(output.glob("*.pdf")) and not (output / "overview_measurements.csv").exists()
        assert not list(output.glob(".*pending*"))
        metadata = json.loads((output / "overview_selection.json").read_text())
        assert metadata["cancelled"] is True and metadata["fields"] == []
        assert metadata["requested_field_count"] == 3 and metadata["written_field_count"] == 0
        assert metadata["measurements"]["path"] is None
        assert metadata["measurements"]["written_field_count"] == 0
        assert browser.tr("Measurements CSV was not published because export was cancelled.") in dialog.status.text()
        assert all(path.read_bytes() == content for path, content in original_bytes.items())
    finally:
        retire(dialog)
        dialog.close()
