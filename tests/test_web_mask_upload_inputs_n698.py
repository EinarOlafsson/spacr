"""Real local image/PDF fixtures; no inference, remote data or GPU."""
import hashlib
import io
import json
import zipfile

import numpy as np
from PIL import Image
import pytest
import tifffile

from tools.web_mask_generator import inputs, results


@pytest.fixture
def ome(tmp_path):
    path = tmp_path / "cells.ome.tif"
    pixels = np.arange(2 * 3 * 2 * 5 * 7, dtype=np.uint16).reshape(2, 3, 2, 5, 7)
    tifffile.imwrite(path, pixels, metadata={"axes": "TZCYX"}, ome=True, photometric="minisblack")
    return path, pixels


def test_explicit_ome_plane_channel_order_and_provenance_preserve_pixels(ome):
    path, pixels = ome
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    description = inputs.describe_upload(path)
    assert description == {"format": "TIFF", "series": [{"shape": [2, 3, 2, 5, 7], "axes": "TZCYX", "dtype": "uint16"}]}
    plane = inputs.read_upload_plane(path, indices={"T": 1, "Z": 2}, channels=[1, 0])
    assert plane["image_axes"] == "CYX"
    np.testing.assert_array_equal(plane["image"], pixels[1, 2, [1, 0]])
    assert plane["image"].dtype == pixels.dtype
    assert plane["selection"] == {"format": "TIFF", "channels": [1, 0], "series": 0, "source_axes": "TZCYX", "indices": {"T": 1, "Z": 2}}
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


@pytest.mark.parametrize("choices", [None, {"T": 0}, {"Z": 1}])
def test_no_implicit_stack_projection_or_time_selection(ome, choices):
    path, _ = ome
    with pytest.raises(ValueError, match="explicit index"):
        inputs.read_upload_plane(path, indices=choices)


def test_series_and_unrecognized_stack_axes_are_explicit(tmp_path):
    path = tmp_path / "series.tif"
    first = np.arange(3 * 4 * 5, dtype=np.uint8).reshape(3, 4, 5)
    second = np.arange(20, dtype=np.uint16).reshape(4, 5) + 1000
    with tifffile.TiffWriter(path) as writer:
        writer.write(first, metadata={"axes": "QYX"}, photometric="minisblack")
        writer.write(second, metadata={"axes": "YX"}, photometric="minisblack")
    assert len(inputs.describe_upload(path)["series"]) == 2
    with pytest.raises(ValueError, match="axis Q"):
        inputs.read_upload_plane(path)
    np.testing.assert_array_equal(inputs.read_upload_plane(path, indices={"Q": 2})["image"], first[2])
    plane = inputs.read_upload_plane(path, series=1)
    assert plane["image_axes"] == "YX"
    np.testing.assert_array_equal(plane["image"], second)


@pytest.mark.parametrize("suffix,mode", [(".png", "L"), (".png", "I;16"), (".png", "RGB"), (".png", "RGBA"), (".jpg", "RGB"), (".bmp", "RGB")])
def test_plain_image_channels_dtype_and_decoded_values(tmp_path, suffix, mode):
    dtype = np.uint16 if mode == "I;16" else np.uint8
    channels = {"L": 1, "I;16": 1, "RGB": 3, "RGBA": 4}[mode]
    shape = (5, 7) if channels == 1 else (5, 7, channels)
    image = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
    path = tmp_path / ("input" + suffix)
    Image.fromarray(image).save(path)
    with Image.open(path) as ordinary:
        decoded = np.array(ordinary)
    plane = inputs.read_upload_plane(path)
    expected = decoded if channels == 1 else np.moveaxis(decoded, -1, 0)
    np.testing.assert_array_equal(plane["image"], expected)
    assert plane["image"].dtype == decoded.dtype
    assert plane["image_axes"] == ("YX" if channels == 1 else "CYX")
    assert plane["selection"]["channels"] == list(range(channels))


def test_real_pdf_pages_render_at_explicit_dpi_without_writing_inputs(tmp_path):
    path = tmp_path / "figures.pdf"
    first = Image.new("RGB", (40, 24), "red")
    second = Image.new("RGB", (30, 20), "blue")
    first.save(path, "PDF", save_all=True, append_images=[second], resolution=72)
    before = path.read_bytes()
    assert inputs.describe_upload(path) == {"format": "PDF", "pages": [{"width_points": 40.0, "height_points": 24.0}, {"width_points": 30.0, "height_points": 20.0}]}
    plane = inputs.read_upload_plane(path, page=1, pdf_dpi=144, channels=[2, 0])
    assert plane["image"].shape == (2, 40, 60)
    assert plane["image_axes"] == "CYX"
    assert plane["image"][0].mean() > 240
    assert plane["image"][1].mean() < 15
    assert plane["selection"] == {"format": "PDF", "channels": [2, 0], "page": 1, "pdf_dpi": 144, "pixels": "PDF page rendering"}
    assert path.read_bytes() == before


@pytest.mark.parametrize("changes", [
    {"indices": {"T": 2, "Z": 0}}, {"indices": {"T": -1, "Z": 0}},
    {"indices": {"T": True, "Z": 0}}, {"indices": {"T": 1.5, "Z": 0}},
    {"indices": {"T": 0, "Z": 0, "X": 1}},
    {"channels": []}, {"channels": [2]}, {"channels": [0, 0]},
    {"channels": [-1]}, {"channels": [True]},
    {"series": 1}, {"series": -1}, {"page": 1},
    {"max_file_bytes": 1}, {"max_file_bytes": 0},
    {"max_decoded_bytes": 1}, {"max_decoded_bytes": 0},
    {"max_plane_pixels": 1}, {"pdf_dpi": 0},
])
def test_bad_selection_or_limits_are_rejected_before_pixel_decode(ome, changes, monkeypatch):
    path, _ = ome
    def no_decode(*args, **kwargs):
        pytest.fail("pixel decode ran before selection/size rejection")
    monkeypatch.setattr(tifffile.TiffPageSeries, "asarray", no_decode)
    arguments = {"indices": {"T": 0, "Z": 0}, **changes}
    with pytest.raises(ValueError):
        inputs.read_upload_plane(path, **arguments)


def test_palette_and_multiframe_plain_images_are_explicitly_refused(tmp_path):
    palette = tmp_path / "palette.png"
    Image.new("P", (4, 4)).save(palette)
    with pytest.raises(ValueError, match="color mode"):
        inputs.describe_upload(palette)
    multi = tmp_path / "multi.png"
    Image.new("RGB", (4, 4), "red").save(multi, save_all=True, append_images=[Image.new("RGB", (4, 4), "blue")])
    with pytest.raises(ValueError, match="Multi-frame"):
        inputs.describe_upload(multi)


def test_unknown_format_missing_file_and_ambiguous_axes_fail(tmp_path):
    path = tmp_path / "sample.unsupported"
    path.write_bytes(b"not a supported upload yet")
    with pytest.raises(ValueError, match="Unsupported"):
        inputs.describe_upload(path)
    with pytest.raises(ValueError, match="missing"):
        inputs.describe_upload(tmp_path / "missing.tif")
    with pytest.raises(ValueError, match="unambiguous"):
        inputs._selection((2, 3, 4), "QYXQ", {}, None, 100)


def test_selected_plane_receipt_can_be_packaged_with_actual_arrays(ome, tmp_path):
    path, _ = ome
    selected = inputs.read_upload_plane(path, indices={"T": 1, "Z": 2}, channels=[0])
    target = tmp_path / "download.zip"
    masks = np.zeros(selected["image"].shape[-2:], dtype=np.uint32)
    masks[1:3, 2:4] = 1
    results.build_result_zip(
        target, input_file=path, image=selected["image"], image_axes=selected["image_axes"],
        masks=masks, model_id="fixture", model_version="one", weights_sha256="12" * 32,
        settings={"input_selection": selected["selection"]},
        unavailable={"flows": "fixture supplies no vectors", "cell_probability": "fixture supplies no scores"})
    with zipfile.ZipFile(target) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["settings"]["input_selection"] == selected["selection"]
        np.testing.assert_array_equal(tifffile.imread(io.BytesIO(archive.read("image.tif"))), selected["image"])


@pytest.fixture
def pdf(tmp_path):
    path = tmp_path / "page.pdf"
    Image.new("RGB", (40, 24), "white").save(path, "PDF", resolution=72)
    return path


@pytest.mark.parametrize("changes", [{"series": 1}, {"indices": {"T": 1}}, {"page": 1}, {"max_decoded_bytes": 1}, {"max_plane_pixels": 1}])
def test_pdf_selection_and_limits_are_checked_before_render(pdf, changes, monkeypatch):
    import pypdfium2
    def no_render(*args, **kwargs):
        pytest.fail("render ran before selection/size rejection")
    monkeypatch.setattr(pypdfium2.PdfPage, "render", no_render)
    with pytest.raises(ValueError):
        inputs.read_upload_plane(pdf, **changes)


def test_pdf_missing_reader_and_page_limit_are_explained(pdf, monkeypatch):
    import sys
    import pypdfium2
    with monkeypatch.context() as patch:
        patch.setattr(inputs, "MAX_PDF_PAGES", 0)
        with pytest.raises(ValueError, match="page limit"):
            inputs.describe_upload(pdf)
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, "pypdfium2", None)
        with pytest.raises(ValueError, match="pypdfium2"):
            inputs.describe_upload(pdf)
