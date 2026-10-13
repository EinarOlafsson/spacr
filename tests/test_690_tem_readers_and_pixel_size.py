"""N690: Make Masks reads TEM files and saves masks with their pixel size.

Gatan DigitalMicrograph files are parsed by spaCR itself (the open readers
are GPL); the files here come from :mod:`tests.tem_files`, whose output an
independent reader (ncempy) returns identically. MRC goes through the
optional ``mrcfile``. Every saved mask is checked at the image's full size
with the pixel size the source recorded.
"""
from __future__ import annotations

import builtins
import sys

import numpy as np
import pytest
import tifffile

from spacr import io as sio
from spacr.qt import mask_engine as engine
from tests.tem_files import write_dm

DTYPES = ("u1", "i1", "<u2", "<i2", "<u4", "<i4", "<f4", "<f8")


@pytest.fixture(autouse=True)
def _forget_pixel_sizes():
    """Each test starts with no pixel size remembered from another."""
    engine._PIXEL_SIZES.clear()
    yield
    engine._PIXEL_SIZES.clear()


@pytest.mark.parametrize("version", (3, 4))
@pytest.mark.parametrize("dtype", DTYPES)
def test_dm_round_trips_every_type(tmp_path, version, dtype):
    rng = np.random.default_rng(690)
    image = (rng.random((37, 53)) * 100).astype(dtype)
    path = tmp_path / f"field.dm{version}"
    write_dm(path, image, version=version, scale=0.37, unit="nm")
    pixels, meta = sio.read_dm(path)
    assert pixels.shape == (37, 53)
    assert pixels.dtype == np.dtype(dtype).newbyteorder("=")
    assert np.array_equal(pixels, image)
    assert meta == {"pixel_size_nm": 0.37, "unit": "nm",
                    "format": f"dm{version}"}


@pytest.mark.parametrize("unit,scale,expected", [
    ("nm", 1.25, 1.25), ("µm", 0.002, 2.0), ("um", 0.5, 500.0),
    ("Å", 4.0, 0.4), ("1/nm", 0.1, None), ("", 1.0, None)])
def test_dm_calibration_becomes_nanometres(tmp_path, unit, scale, expected):
    path = tmp_path / "field.dm4"
    write_dm(path, np.zeros((8, 6), np.uint16), scale=scale, unit=unit)
    _image, meta = sio.read_dm(path)
    assert meta["unit"] == unit
    if expected is None:
        assert meta["pixel_size_nm"] is None
    else:
        assert meta["pixel_size_nm"] == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize("little", (True, False))
def test_dm_long_arrays_tags_and_byte_order(tmp_path, little):
    image = np.arange(120 * 90, dtype="<u2").reshape(120, 90)
    path = tmp_path / "big.dm3"
    write_dm(path, image, version=3, metadata=True, little=little,
             thumbnail=False)
    pixels, _meta = sio.read_dm(path)
    assert pixels.size > sio._DM_EAGER
    assert np.array_equal(pixels, image)
    assert pixels.dtype.byteorder in ("=", "|")


def test_dm_metadata_values_are_parsed(tmp_path):
    path = tmp_path / "tags.dm4"
    write_dm(path, np.ones((3, 4), np.uint8), metadata=True)
    with open(path, "rb") as handle:
        tags = sio._DMReader(handle).group()
    assert tags["Label"] == "hello"
    assert tuple(float(v) for v in tags["Pair"]) == (1.5, 2.5)
    assert tags["Points"]["f0"].tolist() == [1, 3]


def test_dm_refuses_what_make_masks_cannot_draw(tmp_path):
    rgb_only = tmp_path / "rgb.dm4"
    write_dm(rgb_only, np.zeros((4, 4), "<u4"), datatype=23, thumbnail=False)
    with pytest.raises(ValueError, match="no greyscale image"):
        sio.read_dm(rgb_only)
    stack = tmp_path / "stack.dm4"
    write_dm(stack, np.zeros((6, 4), np.uint16), dims=(4, 3, 2))
    with pytest.raises(ValueError, match="one 2-D image"):
        sio.read_dm(stack)
    malformed = tmp_path / "malformed.dm3"
    write_dm(malformed, np.zeros((2, 2), np.uint8), version=3, thumbnail=False)
    raw = bytearray(malformed.read_bytes())
    marker = raw.index(b"%%%%", raw.index(b"Data"))
    raw[marker + 8:marker + 20] = (np.array([20, 15, 0], ">u4").tobytes())
    malformed.write_bytes(bytes(raw))
    with pytest.raises(ValueError, match="malformed"):
        sio.read_dm(malformed)


def test_dm_malformedimage_is_refused_by_name(tmp_path, monkeypatch):
    path = tmp_path / "fft.dm4"
    write_dm(path, np.zeros((2, 2), np.uint8))
    real_value = sio._DMReader.value

    def as_complex(self):
        out = real_value(self)
        if isinstance(out, np.ndarray) and out.size == 4 and out.dtype == np.uint8:
            return np.zeros(4, np.dtype([("re", "<f4"), ("im", "<f4")]))
        return out

    monkeypatch.setattr(sio._DMReader, "value", as_complex)
    with pytest.raises(ValueError, match="complex"):
        sio.read_dm(path)


@pytest.mark.parametrize("payload,message", [
    (b"\x00\x00\x00\x05" + b"\x00" * 12, "version 5"),
    (b"\x00\x00\x00\x04\x00", "ends inside"),
])
def test_dm_refuses_a_file_that_is_not_one(tmp_path, payload, message):
    path = tmp_path / "bad.dm4"
    path.write_bytes(payload)
    with pytest.raises(ValueError, match=message):
        sio.read_dm(path)


def test_dm_truncated_and_malformed_tags(tmp_path):
    path = tmp_path / "field.dm4"
    write_dm(path, np.arange(16, dtype=np.uint8).reshape(4, 4), thumbnail=False)
    whole = path.read_bytes()
    cut = tmp_path / "cut.dm4"
    cut.write_bytes(whole[:-40])
    with pytest.raises(ValueError, match="ends inside"):
        sio.read_dm(cut)
    raw = bytearray(whole)
    raw[raw.index(b"%%%%")] = ord("x")
    broken = tmp_path / "broken.dm4"
    broken.write_bytes(bytes(raw))
    with pytest.raises(ValueError, match="marker"):
        sio.read_dm(broken)
    raw = bytearray(whole)
    raw[16 + 2 + 8] = 7
    odd = tmp_path / "odd.dm4"
    odd.write_bytes(bytes(raw))
    with pytest.raises(ValueError, match="tag kind 7"):
        sio.read_dm(odd)
    with open(path, "rb") as handle:
        reader = sio._DMReader(handle)
        with pytest.raises(ValueError, match="unsupported"):
            reader._dtype(99)


def test_dm_text_and_units():
    assert sio._dm_text("nm") == "nm"
    assert sio._dm_text(np.array([110, 109, 0], np.uint16)) == "nm"
    assert sio._dm_text(3.0) == ""
    assert sio._nm_per("µm") == 1000.0
    assert sio._nm_per(None) is None


def _write_mrc(path, data, voxel):
    mrcfile = pytest.importorskip("mrcfile")
    with mrcfile.new(path, overwrite=True) as mrc:
        mrc.set_data(data)
        mrc.voxel_size = voxel


@pytest.mark.parametrize("shape", ((20, 30), (1, 20, 30)))
def test_mrc_round_trip_with_pixel_size(tmp_path, shape):
    data = np.arange(np.prod(shape), dtype=np.int16).reshape(shape)
    path = tmp_path / "section.mrc"
    _write_mrc(path, data, 12.5)
    image, meta = sio.read_mrc(path)
    assert image.shape == (20, 30)
    assert np.array_equal(image, data.reshape(20, 30))
    assert meta["pixel_size_nm"] == pytest.approx(1.25)


def test_mrc_without_voxel_size_and_stacks(tmp_path):
    flat = tmp_path / "flat.rec"
    _write_mrc(flat, np.zeros((5, 6), np.float32), 0.0)
    assert sio.read_mrc(flat)[1]["pixel_size_nm"] is None
    stack = tmp_path / "stack.mrc"
    _write_mrc(stack, np.zeros((3, 5, 6), np.float32), 1.0)
    with pytest.raises(ValueError, match="save one section"):
        sio.read_mrc(stack)


def test_missing_mrcfile_names_the_extra(tmp_path, monkeypatch):
    real_import = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name == "mrcfile":
            raise ImportError("No module named 'mrcfile'")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "mrcfile", raising=False)
    monkeypatch.setattr(builtins, "__import__", refuse)
    with pytest.raises(sio.MissingTEMReader) as caught:
        sio.read_mrc(tmp_path / "x.MRC")
    assert caught.value.kind == ".mrc" and caught.value.package == "mrcfile"
    assert 'pip install "spacr[tem]"' in str(caught.value)
    assert sio.image_pixel_size_nm(tmp_path / "x.mrc") is None


def test_tiff_pixel_size_from_every_record(tmp_path):
    plain = tmp_path / "plain.tif"
    tifffile.imwrite(plain, np.zeros((4, 4), np.uint8))
    assert sio.tiff_pixel_size_nm(plain) is None
    inch = tmp_path / "inch.tif"
    tifffile.imwrite(inch, np.zeros((4, 4), np.uint8), resolution=(72, 72),
                     resolutionunit="INCH")
    assert sio.tiff_pixel_size_nm(inch) is None
    cm = tmp_path / "cm.tif"
    tifffile.imwrite(cm, np.zeros((4, 4), np.uint8), resolution=(5e6, 5e6),
                     resolutionunit="CENTIMETER")
    assert sio.tiff_pixel_size_nm(cm) == pytest.approx(2.0)
    imagej = tmp_path / "imagej.tif"
    tifffile.imwrite(imagej, np.zeros((4, 4), np.uint8), imagej=True,
                     resolution=(250.0, 250.0), metadata={"unit": "um"})
    assert sio.tiff_pixel_size_nm(imagej) == pytest.approx(4.0)
    zero = tmp_path / "zero.tif"
    tifffile.imwrite(zero, np.zeros((4, 4), np.uint8), resolution=((0, 1), (0, 1)),
                     resolutionunit="CENTIMETER")
    assert sio.tiff_pixel_size_nm(zero) is None
    junk = tmp_path / "junk.tif"
    junk.write_bytes(b"not a tiff")
    assert sio.tiff_pixel_size_nm(junk) is None


def test_tiff_without_resolution_tags(tmp_path, monkeypatch):
    path = tmp_path / "x.tif"
    tifffile.imwrite(path, np.zeros((4, 4), np.uint8))

    class _Tags(dict):
        def get(self, key, default=None):
            return None

    class _Page:
        tags = _Tags()

    class _Tif:
        pages = [_Page()]
        imagej_metadata = None
        shaped_metadata = None

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(sio.tifffile, "TiffFile", lambda _p: _Tif())
    assert sio.tiff_pixel_size_nm(path) is None


def test_read_tem_image_and_pixel_size_dispatch(tmp_path):
    dm = tmp_path / "a.dm3"
    write_dm(dm, np.ones((5, 7), np.uint16), version=3, scale=3.0)
    assert sio.read_tem_image(dm)[1]["pixel_size_nm"] == pytest.approx(3.0)
    tif = tmp_path / "b.tif"
    tifffile.imwrite(tif, np.ones((5, 7), np.uint16), resolution=(1e6, 1e6),
                     resolutionunit="CENTIMETER")
    image, meta = sio.read_tem_image(tif)
    assert image.shape == (5, 7) and meta["pixel_size_nm"] == pytest.approx(10.0)
    assert sio.image_pixel_size_nm(tif) == pytest.approx(10.0)
    assert sio.image_pixel_size_nm(dm) == pytest.approx(3.0)
    assert sio.image_pixel_size_nm(tmp_path / "c.png") is None
    assert sio.image_pixel_size_nm(tmp_path / "missing.dm4") is None


def test_make_masks_lists_reads_and_remembers_tem_files(tmp_path):
    image = (np.arange(48 * 64) % 251).astype(np.uint16).reshape(48, 64)
    write_dm(tmp_path / "f1.dm4", image, scale=0.85)
    write_dm(tmp_path / "f2.dm3", image, version=3, scale=2.0)
    assert engine.list_images(str(tmp_path)) == ["f1.dm4", "f2.dm3"]
    loaded, mask = engine.load_image_and_mask(str(tmp_path), "f1.dm4")
    assert loaded.shape == mask.shape == (48, 64)
    key = str((tmp_path / "f1.dm4").resolve())
    assert engine._PIXEL_SIZES[key] == pytest.approx(0.85)
    assert engine.pixel_size_nm(tmp_path / "f1.dm4") == pytest.approx(0.85)
    assert engine.pixel_size_nm(tmp_path / "f2.dm3") == pytest.approx(2.0)
    assert str((tmp_path / "f2.dm3").resolve()) in engine._PIXEL_SIZES
    assert engine.pixel_size_nm(tmp_path / "none.png") is None


def test_saved_mask_is_full_size_with_the_pixel_size(tmp_path):
    image = np.zeros((300, 410), np.uint16)
    write_dm(tmp_path / "field.dm4", image, scale=0.85)
    engine.load_image_and_mask(str(tmp_path), "field.dm4")
    mask = np.zeros((300, 410), np.uint16)
    mask[10:50, 20:60] = 1
    mask[100:150, 200:260] = 2
    path = engine.save_mask(str(tmp_path), "field.dm4", mask)
    assert path.endswith("field.tif")
    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        assert page.shape == (300, 410)
        assert page.tags["ResolutionUnit"].value == 3
        numerator, denominator = page.tags["XResolution"].value
    assert 1e7 * denominator / numerator == pytest.approx(0.85, rel=1e-8)
    assert sio.tiff_pixel_size_nm(path) == 0.85
    assert sorted(np.unique(tifffile.imread(path)).tolist()) == [0, 1, 2]


def test_saved_mask_of_an_uncalibrated_image_has_no_pixel_size(tmp_path):
    tifffile.imwrite(tmp_path / "plain.tif", np.zeros((20, 30), np.uint16))
    mask = np.zeros((20, 30), np.uint8)
    mask[2:5, 2:5] = 1
    path = engine.save_mask(str(tmp_path), "plain.tif", mask)
    assert sio.tiff_pixel_size_nm(path) is None
    assert engine._resolution_for(None) == {}
    assert engine._resolution_for(-1.0) == {}
    huge = engine._resolution_for(1e8)
    assert huge["metadata"] == {"pixel_size_nm": 1e8}


def test_contrast_presets_are_complete():
    for key in ("tem", "tem_inverted"):
        preset = engine.CONTRAST_PRESETS[key]
        assert set(preset) == {"lower", "upper", "invert", "clahe",
                               "clahe_tile", "clahe_clip"}
        assert 0 <= preset["lower"] < preset["upper"] <= 100
    assert engine.CONTRAST_PRESETS["tem_inverted"]["invert"] is True


def test_stretching_a_crop_matches_cropping_the_stretch():
    rng = np.random.default_rng(7)
    image = rng.integers(0, 4096, (90, 120), dtype=np.uint16)
    whole = engine.normalize_uint16(image, 1.0, 99.9)
    lo, hi = engine._percentile_bounds(image, 1.0, 99.9)
    crop = engine._stretch_between(image[10:70:3, 5:100:3], lo, hi)
    assert np.array_equal(crop, whole[10:70:3, 5:100:3])
    flat = np.full((4, 4), 9, np.uint16)
    assert engine._percentile_bounds(flat, 1, 99) == (9.0, 10.0)
    assert engine.normalize_uint16(np.zeros((0,), np.uint16)).size == 0


@pytest.mark.parametrize("size,target,step", [
    ((8192, 8192), (1000, 500), 8), ((1024, 1024), (1000, 1000), 1),
    ((500, 400), (1000, 800), 1), ((4096, 1024), (1000, 0), 4)])
def test_display_step(size, target, step):
    assert engine._display_step(*size, *target) == step


def test_tem_extensions_are_one_list_everywhere():
    from spacr.curation_queue import IMAGE_EXTS as queue_exts

    assert queue_exts == engine.IMAGE_EXTS
    for suffix in sio.TEM_EXTS:
        assert suffix in engine.IMAGE_EXTS
    assert set(sio.TEM_EXTS) == set(engine.TEM_SUFFIXES)


def test_tem_example_set_is_registered():
    from spacr import example_archives as ea

    example = ea.example_set("tem")
    assert example.repo == ea.TEM_EXAMPLE_REPO == "einarolafsson/spacr-example-tem"
    assert example.archive == "spacr-example-tem.tar"
    assert example.folder == "make_masks_tem" and not example.in_default
