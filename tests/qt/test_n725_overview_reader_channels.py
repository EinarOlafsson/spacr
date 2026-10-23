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
        captured.append((np.asarray(figure.axes[0].images[0].get_array()).copy(),
                         figure._spacr_spec))
        return original(figure, *args, **kwargs)

    monkeypatch.setattr(plot, "save_figure", save)
    output = tmp_path / "export"
    written, errors = browser._overview_export((field,), output, tuple(range(channel_count)), (1, 99),
        measurements=True, quality_edge=None, dpi=150, conditions={(field.plate, field.well): "control"})
    assert len(written) == 1 and not errors
    drawn = captured[0][0]
    if channel_count == 1:
        assert drawn.shape == image.shape + (3,)
    else:
        tile = max(image.shape) + 16
        assert drawn.shape == (tile * 2, tile * 2, 3)
        for index in range(4):
            plane = stored[index].astype(np.float32)
            low, high = np.percentile(plane, (1, 99))
            expected = np.uint8(np.clip((plane - low) / (high - low), 0, 1) * 255)
            y, x = (index // 2) * tile + 16, (index % 2) * tile
            np.testing.assert_array_equal(drawn[y:y + image.shape[0], x:x + image.shape[1], 0], expected)
    assert captured[0][1]["original_resolution"] is True
    from pypdf import PdfReader
    embedded = PdfReader(written[0]).pages[0].images
    assert len(embedded) == 1
    assert embedded[0].image.size == (drawn.shape[1], drawn.shape[0])
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
