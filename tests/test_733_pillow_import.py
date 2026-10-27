"""Installed raster readers share header preview and lossless Import routes."""
import numpy as np
import pytest
import tifffile
from PIL import Image, features

from spacr import convert as cv


@pytest.mark.parametrize("suffix", [".gif", ".webp", ".ppm"])
def test_installed_raster_reader_roundtrip_preserves_channels_and_source(tmp_path, suffix):
    if suffix == ".webp" and not features.check("webp"):
        pytest.skip("This Pillow has no WebP reader")
    source = tmp_path / "source"
    source.mkdir()
    path = source / ("field" + suffix)
    pixels = np.zeros((5, 7, 3), np.uint8)
    pixels[:, :3] = [255, 0, 0]
    pixels[:, 3:] = [0, 255, 0]
    Image.fromarray(pixels).save(path, **({"lossless": True} if suffix == ".webp" else {}))
    original = path.read_bytes()
    assert suffix in cv._image_extensions()
    sources = cv.scan(str(source))
    assert len(sources) == 1 and sources[0].readable
    assert sources[0].n_channels == 3
    mapped = cv._read_source(sources[0])
    np.testing.assert_array_equal(mapped[0, 0], np.moveaxis(pixels, -1, 0))
    result = cv.convert(cv.plan(sources), str(tmp_path / "converted"))
    assert len(result.written) == 3
    for row in result.written:
        np.testing.assert_array_equal(tifffile.imread(tmp_path / "converted" / row.target),
                                      pixels[:, :, row.channel - 1])
    assert path.read_bytes() == original


def test_raster_header_preview_does_not_decode_pixels(tmp_path, monkeypatch):
    Image.new("RGB", (7, 5), "red").save(tmp_path / "field.gif")
    monkeypatch.setattr(Image.Image, "load", lambda *args: pytest.fail("Preview decoded pixels"))
    source = cv.scan(str(tmp_path))[0]
    assert source.readable and source.meta["shape"] == (5, 7, 3)


def test_transparent_palette_keeps_alpha_as_a_channel(tmp_path):
    image = Image.new("P", (3, 2))
    image.putpalette([255, 0, 0, 0, 255, 0] + [0] * 762)
    image.putdata([0, 1, 0, 1, 0, 1])
    path = tmp_path / "field.gif"
    image.save(path, transparency=0)
    source = cv.scan(str(tmp_path))[0]
    assert source.n_channels == 4
    with Image.open(path) as handle:
        expected = np.asarray(handle.convert("RGBA"))
    np.testing.assert_array_equal(cv._read_source(source)[0, 0], np.moveaxis(expected, -1, 0))


def test_multiframe_raster_is_refused_instead_of_first_frame(tmp_path):
    first = Image.new("RGB", (7, 5), "red")
    second = Image.new("RGB", (7, 5), "blue")
    path = tmp_path / "movie.gif"
    first.save(path, save_all=True, append_images=[second])
    original = path.read_bytes()
    source = cv.scan(str(tmp_path))[0]
    assert not source.readable
    assert "explicit frame or axis selection" in source.error
    assert not cv.plan([source]).mappings
    assert path.read_bytes() == original


def test_changed_raster_shape_is_refused_at_actual_read(tmp_path):
    path = tmp_path / "field.gif"
    Image.new("RGB", (7, 5), "red").save(path)
    source = cv.scan(str(tmp_path))[0]
    Image.new("RGB", (8, 5), "red").save(path)
    with pytest.raises(cv.ConfigurationError, match="changed after the import preview"):
        cv._read_source(source)


def test_multiple_icon_resolutions_are_not_silently_selected(tmp_path):
    Image.new("RGB", (32, 32), "red").save(tmp_path / "multiple.ico", sizes=[(16, 16), (32, 32)])
    source = cv.scan(str(tmp_path))[0]
    assert not source.readable and "explicit frame or axis selection" in source.error


def test_dynamic_pillow_registry_and_scientific_tiff_priority(tmp_path, monkeypatch):
    Image.init()
    monkeypatch.setitem(Image.EXTENSION, ".spacr_raster_fixture", "PNG")
    cv._pillow_extensions.cache_clear()
    try:
        path = tmp_path / "field.spacr_raster_fixture"
        Image.new("L", (7, 5), 12).save(path, format="PNG")
        assert path.suffix in cv._image_extensions()
        source = cv.scan(str(tmp_path))[0]
        assert source.readable
        np.testing.assert_array_equal(cv._read_source(source)[0, 0, 0], np.full((5, 7), 12))
        monkeypatch.setattr(cv, "_describe_plain", lambda path: pytest.fail("TIFF lost its scientific reader"))
        tiff = tmp_path / "volume.tif"
        tifffile.imwrite(tiff, np.ones((2, 5, 7), np.uint16), metadata={"axes": "ZYX"})
        described = cv._describe(str(tiff), ".tif")
        assert described["reader"] == "tifffile" and described["n_z"] == 2
    finally:
        cv._pillow_extensions.cache_clear()
