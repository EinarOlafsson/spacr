"""The pixel size a TIFF records, read with tifffile alone."""

from types import SimpleNamespace

import numpy as np
import pytest
import tifffile

from spacr import tiff_io


def _tag(numerator, denominator):
    return SimpleNamespace(value=(numerator, denominator))


def _unit(value):
    return SimpleNamespace(value=value)


@pytest.mark.parametrize("unit, nanometres", [
    ("nm", 1.0), (" µm ", 1000.0), ("um", 1000.0), ("Å", 0.1),
    ("mm", 1e6), (None, None), ("", None), ("furlong", None)])
def test_a_length_unit_is_nanometres_or_nothing(unit, nanometres):
    assert tiff_io._nm_per(unit) == nanometres


def test_an_unreadable_tiff_has_no_calibration(tmp_path):
    (tmp_path / "broken.tif").write_bytes(b"not a tiff")
    assert tiff_io._tiff_calibration_tags(tmp_path / "broken.tif") is None
    assert tiff_io._tiff_pixel_size_nm(tmp_path / "broken.tif") is None
    assert tiff_io._tiff_pixel_size_nm(tmp_path / "missing.tif") is None


def test_a_written_pixel_size_in_nm_is_read_back(tmp_path):
    path = tmp_path / "exact.tif"
    tifffile.imwrite(path, np.zeros((4, 4), np.uint16),
                     metadata={"pixel_size_nm": 12.5})
    assert tiff_io._tiff_pixel_size_nm(path) == 12.5


def test_an_imagej_unit_and_resolution_give_the_pixel_size(tmp_path):
    path = tmp_path / "imagej.tif"
    tifffile.imwrite(path, np.zeros((4, 4), np.uint16), imagej=True,
                     resolution=(4.0, 4.0), metadata={"unit": "um"})
    assert tiff_io._tiff_pixel_size_nm(path) == pytest.approx(250.0)


def test_a_centimetre_resolution_unit_gives_the_pixel_size(tmp_path):
    path = tmp_path / "cm.tif"
    tifffile.imwrite(path, np.zeros((4, 4), np.uint16),
                     resolution=(1e5, 1e5), resolutionunit="CENTIMETER",
                     metadata=None)
    assert tiff_io._tiff_pixel_size_nm(path) == pytest.approx(100.0)


def test_a_tiff_with_no_resolution_has_no_pixel_size(tmp_path):
    path = tmp_path / "plain.tif"
    tifffile.imwrite(path, np.zeros((4, 4), np.uint16), metadata=None)
    assert tiff_io._tiff_pixel_size_nm(path) is None


@pytest.mark.parametrize("shaped", [{"pixel_size_nm": 0},
                                    {"pixel_size_nm": "9"},
                                    ["not", "a", "mapping"], {}])
def test_an_unusable_declared_size_falls_through_to_the_resolution_tag(
        monkeypatch, shaped):
    monkeypatch.setattr(
        tiff_io, "_tiff_calibration_tags",
        lambda path: (_tag(2, 1), None, {"unit": "um"}, shaped))
    assert tiff_io._tiff_pixel_size_nm("x.tif") == pytest.approx(500.0)


def test_a_missing_tag_has_no_size(monkeypatch):
    monkeypatch.setattr(tiff_io, "_tiff_calibration_tags",
                        lambda path: (None, None, {}, {}))
    assert tiff_io._tiff_pixel_size_nm("x.tif") is None


@pytest.mark.parametrize("tag", [_tag(0, 1), _tag(1, 0)])
def test_a_resolution_with_a_zero_part_has_no_size(tag):
    assert tiff_io._resolution_tag_nm(tag, None, {"unit": "nm"}) is None


def test_a_negative_resolution_has_no_size():
    assert tiff_io._resolution_tag_nm(
        _tag(-2, 1), None, {"unit": "nm"}) is None


@pytest.mark.parametrize("unit_tag", [None, _unit(1), _unit(2)])
def test_an_unknown_length_unit_without_centimetres_has_no_size(unit_tag):
    assert tiff_io._resolution_tag_nm(_tag(1, 1), unit_tag, {}) is None


def test_centimetres_from_the_resolution_unit_are_used_without_imagej():
    assert tiff_io._resolution_tag_nm(_tag(10, 1), _unit(3), {}) == 1e6
