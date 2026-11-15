"""The pixel size a TIFF records, read with tifffile alone (``spacr.tiff_io``).

``_tiff_pixel_size_nm`` prefers an explicit ``pixel_size_nm`` in the shaped
metadata, falls back to the XResolution tag (unit from ImageJ metadata or the
ResolutionUnit tag), and answers ``None`` whenever the file or the tags cannot
give an honest size.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import tifffile

from spacr import tiff_io


def _tag(numerator, denominator):
    return SimpleNamespace(value=(numerator, denominator))


def _unit(value):
    return SimpleNamespace(value=value)


def _write(path, **kwargs):
    tifffile.imwrite(path, np.zeros((4, 4), np.uint16), **kwargs)
    return path


def test_a_tiff_pixel_size_comes_from_the_declared_size_or_the_resolution_tag(
        subtests, tmp_path, monkeypatch):
    """Written files and stubbed tag sets give their exact size or None."""
    files = [
        ("not a tiff", lambda d: (d / "broken.tif").write_bytes(b"not a tiff")
         and d / "broken.tif", None),
        ("missing file", lambda d: d / "missing.tif", None),
        ("pixel_size_nm in metadata", lambda d: _write(
            d / "exact.tif", metadata={"pixel_size_nm": 12.5}), 12.5),
        ("imagej unit um and 4 px per unit", lambda d: _write(
            d / "imagej.tif", imagej=True, resolution=(4.0, 4.0),
            metadata={"unit": "um"}), 250.0),
        ("centimetre resolution unit", lambda d: _write(
            d / "cm.tif", resolution=(1e5, 1e5), resolutionunit="CENTIMETER",
            metadata=None), 100.0),
        ("no resolution at all", lambda d: _write(d / "plain.tif",
                                                  metadata=None), None),
    ]
    for case_id, make, expected in files:
        with subtests.test(msg=case_id):
            path = make(tmp_path)
            got = tiff_io._tiff_pixel_size_nm(path)
            if expected is None:
                assert got is None
            else:
                assert got == pytest.approx(expected)

    with subtests.test(msg="unreadable file has no calibration tags"):
        assert tiff_io._tiff_calibration_tags(tmp_path / "broken.tif") is None

    # An unusable declared size falls through to the resolution tag (2 px/um).
    stubbed = [
        ("declared zero", (_tag(2, 1), None, {"unit": "um"},
                           {"pixel_size_nm": 0}), 500.0),
        ("declared text", (_tag(2, 1), None, {"unit": "um"},
                           {"pixel_size_nm": "9"}), 500.0),
        ("shaped not a mapping", (_tag(2, 1), None, {"unit": "um"},
                                  ["not", "a", "mapping"]), 500.0),
        ("shaped empty", (_tag(2, 1), None, {"unit": "um"}, {}), 500.0),
        ("declared int wins", (_tag(2, 1), None, {"unit": "um"},
                               {"pixel_size_nm": 7}), 7.0),
        ("no resolution tag", (None, None, {}, {}), None),
    ]
    for case_id, tags, expected in stubbed:
        with subtests.test(msg=case_id):
            monkeypatch.setattr(tiff_io, "_tiff_calibration_tags",
                                lambda path, tags=tags: tags)
            assert tiff_io._tiff_pixel_size_nm("x.tif") == expected


def test_a_resolution_tag_gives_nanometres_per_pixel_or_nothing(subtests):
    """Zero, negative and unit-less resolutions give None; cm gives 1e7/res."""
    cases = [
        ("no tag", None, None, {"unit": "nm"}, None),
        ("zero numerator", _tag(0, 1), None, {"unit": "nm"}, None),
        ("zero denominator", _tag(1, 0), None, {"unit": "nm"}, None),
        ("negative resolution", _tag(-2, 1), None, {"unit": "nm"}, None),
        ("no unit anywhere", _tag(1, 1), None, {}, None),
        ("resolution unit none (1)", _tag(1, 1), _unit(1), {}, None),
        ("resolution unit inch (2)", _tag(1, 1), _unit(2), {}, None),
        ("resolution unit cm (3)", _tag(10, 1), _unit(3), {}, 1e6),
        ("imagej nm beats the unit tag", _tag(4, 1), _unit(3),
         {"unit": "nm"}, 0.25),
        ("imagej um with a fraction", _tag(1, 2), None, {"unit": "um"},
         2000.0),
    ]
    for case_id, tag, unit_tag, imagej, expected in cases:
        with subtests.test(msg=case_id):
            assert tiff_io._resolution_tag_nm(tag, unit_tag, imagej) == expected
